import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import numpy as np
from io import BytesIO

# =====================================================================
# CONFIGURAÇÃO DA PÁGINA DO STREAMLIT
# =====================================================================
st.set_page_config(page_title="Dashboard de Cobrança", layout="wide")

# =====================================================================
# BARRA LATERAL: PARÂMETROS DA ANÁLISE
# =====================================================================
st.sidebar.header("⚙️ Parâmetros da Análise")
st.sidebar.write("Ajuste as regras de negócio antes de processar os dados.")

limite_dias_pagamento = st.sidebar.number_input(
    "Janela de Pagamento (dias)", 
    min_value=0, 
    value=10, 
    step=1,
    help="Considerar apenas pagamentos realizados até X dias após a data do contato."
)

limite_tempo_duracao = st.sidebar.number_input(
    "Duração Mínima da Ligação (segundos)", 
    min_value=0, 
    value=60, 
    step=1,
    help="Tempo mínimo para que tabulações de exceção (ex: Ligação Caiu) sejam consideradas Positivas."
)

# =====================================================================
# INTERFACE PRINCIPAL
# =====================================================================
st.title("📊 Painel de Resultados de Cobrança")
st.write("Faça o upload das bases de dados abaixo para gerar a análise de desempenho e o relatório consolidado.")

# 1. UPLOAD DE ARQUIVOS
col1, col2 = st.columns(2)
with col1:
    arquivo_parquet = st.file_uploader("1. Base de Pagamentos (.parquet)", type=['parquet'])
with col2:
    arquivo_excel = st.file_uploader("2. Base de Contatos (.xlsx)", type=['xlsx'])

# Botão para iniciar o processamento
if arquivo_parquet is not None and arquivo_excel is not None:
    if st.button("🚀 Processar Dados e Gerar Relatórios", type="primary"):

        with st.spinner('Lendo arquivos e processando regras de negócio...'):

            # Lendo os arquivos carregados pelo usuário
            df_pagamentos = pq.read_table(arquivo_parquet).to_pandas()

            # =====================================================================
            # NOVO: TRAVA CONTRA DUPLICAÇÃO DE PAGAMENTOS
            # =====================================================================
            # 1. Remove duplicatas exatas que já possam vir no arquivo original
            df_pagamentos = df_pagamentos.drop_duplicates() 
            # 2. Cria um ID único temporário para cada linha de pagamento
            df_pagamentos['ID_PAGAMENTO_TEMP'] = range(len(df_pagamentos))

            # Trava de segurança para a aba correta e pulando linhas se necessário
            df_contatos = pd.read_excel(arquivo_excel)

            # Validação da coluna 'matricula'
            nome_coluna_matricula = 'Código Imóvel' 
            if nome_coluna_matricula not in df_contatos.columns:
                st.error(f"🚨 Erro: A coluna '{nome_coluna_matricula}' não foi encontrada no arquivo Excel de Contatos.")
                st.warning(f"As colunas que o App encontrou no seu arquivo foram: {', '.join(df_contatos.columns)}")
                st.info("Por favor, corrija o cabeçalho no arquivo Excel e faça o upload novamente.")
                st.stop()

            # Converter colunas de chave para string
            df_pagamentos['MATRICULA_PAGAMENTO'] = df_pagamentos['MATRICULA_PAGAMENTO'].astype(str)
            df_contatos['matricula'] = df_contatos[nome_coluna_matricula].astype(str) # Ajustado para usar a variável correta
            df_contatos['Contrato'] = df_contatos['Contrato'].astype(str)

            # =====================================================================
            # TRATAMENTO DE DUPLICATAS NA BASE DE CONTATOS
            # =====================================================================
            df_contatos['Chave_Cliente'] = df_contatos['matricula'] + "_" + df_contatos['Contrato']

            df_contagem_contatos = df_contatos.groupby(['matricula', 'Contrato', 'Chave_Cliente']).size().reset_index(name='Quantidade_Ligacoes')
            df_contagem_contatos = df_contagem_contatos.sort_values('Quantidade_Ligacoes', ascending=False)

            df_contatos['Data'] = pd.to_datetime(df_contatos['Data'], dayfirst=True)
            df_contatos = df_contatos.sort_values('Data').drop_duplicates(subset='Chave_Cliente', keep='first')

            # =====================================================================
            # CRUZAMENTO DE DADOS (MERGE)
            # =====================================================================
            df_cruzado = pd.merge(
                df_contatos,
                df_pagamentos,
                left_on='matricula',
                right_on='MATRICULA_PAGAMENTO',
                how='inner'
            )

            # =====================================================================
            # NOVO: LIMPANDO OS PAGAMENTOS MULTIPLICADOS PELO MERGE
            # =====================================================================
            # Ordena pela data do contato (para dar o crédito à primeira ligação)
            df_cruzado = df_cruzado.sort_values('Data')
            # Remove qualquer pagamento que tenha sido duplicado por causa de múltiplos contratos
            df_cruzado = df_cruzado.drop_duplicates(subset='ID_PAGAMENTO_TEMP', keep='first')

            # =====================================================================
            # REGRAS DE NEGÓCIO (Usando os parâmetros da barra lateral)
            # =====================================================================
            df_cruzado['DATA_PAGAMENTO'] = pd.to_datetime(df_cruzado['DATA_PAGAMENTO'], dayfirst=True)
            df_cruzado['Dias_Ate_Pagamento'] = (df_cruzado['DATA_PAGAMENTO'] - df_cruzado['Data']).dt.days
            df_cruzado = df_cruzado[(df_cruzado['Dias_Ate_Pagamento'] >= 0) & (df_cruzado['Dias_Ate_Pagamento'] <= limite_dias_pagamento)]

            coluna_duracao = 'Duração'
            df_cruzado['Duracao_Segundos'] = pd.to_timedelta(df_cruzado[coluna_duracao].astype(str)).dt.total_seconds()

            df_cruzado['Conta_Volume'] = np.where(df_cruzado['TIPO_FATURA'] == '1-NOTA FISCAL MENSAL', 1, 0)

            # =====================================================================
            # CATEGORIZAÇÃO DAS TABULAÇÕES
            # =====================================================================
            tab_positivas = ['2a Via de fatura', 'Contato Realizado Com Promessa De Pagamento', 'Negociado Parcelado', 'Telefone Não Pertence Ao Contato', 'Caixa postal', 'Cliente Informa Que Ja Pagou', 'Ligacao Caiu', 'Ligacao Encerrada Pelo Cliente', 'SSP - Inquilino', 'ININ-WRAP-UP-TIMEOUT', 'Ligacao atendida por terceiro', 'S/negociacao - Contato posterior', 'ININ-OUTBOUND-CONTACT-ATTEMPT-LIMIT-SKIPPED', 'SPP- Sem dinheiro', 'SPP - não consegue pagar a vista', 'Usuário cadastro falecido']
            tab_negativas = [   'Ligacao Muda', 'OUT_Posvoice', 'ININ-OUTBOUND-PREVIEW-SKIPPED', 'S/negociação - Fatura Bloqueada', 'S/negociação - Retificação de Fatura', 'Ocupado']
            tabs_excecao_tempo = ['Ligacao Caiu', 'Ligacao Encerrada Pelo Cliente']

            condicao_positiva = (df_cruzado['Tab'].isin(tab_positivas) | (df_cruzado['Tab'].isin(tabs_excecao_tempo) & (df_cruzado['Duracao_Segundos'] >= limite_tempo_duracao)))
            condicao_negativa = (df_cruzado['Tab'].isin(tab_negativas) & ~condicao_positiva)

            df_cruzado['Categoria'] = np.select([condicao_positiva, condicao_negativa], ['Positivas', 'Negativas'], default='Outras')

            mascara_excecao = df_cruzado['Tab'].isin(tabs_excecao_tempo)
            mascara_tempo_maior = df_cruzado['Duracao_Segundos'] >= limite_tempo_duracao

            df_cruzado.loc[mascara_excecao & mascara_tempo_maior, 'Tab'] = df_cruzado['Tab'] + f" (>= {limite_tempo_duracao}s)"
            df_cruzado.loc[mascara_excecao & ~mascara_tempo_maior, 'Tab'] = df_cruzado['Tab'] + f" (< {limite_tempo_duracao}s)"

            df_positivas = df_cruzado[df_cruzado['Categoria'] == 'Positivas']
            df_negativas = df_cruzado[df_cruzado['Categoria'] == 'Negativas']
            df_outras = df_cruzado[df_cruzado['Categoria'] == 'Outras']

        st.success("✅ Dados processados com sucesso! Gerando visualizações...")

        # =====================================================================
        # FUNÇÃO DE ANÁLISE E GRÁFICOS
        # =====================================================================
        def analisar_bloco(df_bloco, nome_bloco, paleta_cores):
            if df_bloco.empty:
                st.warning(f"[{nome_bloco}] Não há dados para exibir neste bloco.")
                return

            valor_arrecadado = df_bloco['VALOR_PAGO'].sum()
            total_pagamentos = df_bloco['Conta_Volume'].sum()

            sns.set_theme(style="whitegrid")
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])

            fig.suptitle(f'Análise de Desempenho - {nome_bloco}\n(Janela de Pagamento: até {limite_dias_pagamento} dias após o contato)', fontsize=16, fontweight='bold')

            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[0, 2])

            agrupado_cidade = df_bloco.groupby('CIDADE')['VALOR_PAGO'].sum().reset_index().sort_values(by='VALOR_PAGO', ascending=False).head(10)
            sns.barplot(data=agrupado_cidade, x='VALOR_PAGO', y='CIDADE', ax=ax0, palette=paleta_cores)
            ax0.set_title('Top 10 Cidades por Valor Arrecadado')
            ax0.set_xlabel('Valor Total Pago (R$)')
            ax0.set_ylabel('')

            df_volume = df_bloco[df_bloco['Conta_Volume'] == 1]

            if not df_volume.empty:
                sns.countplot(data=df_volume, x='TIPO_PAGAMENTO', ax=ax1, palette=paleta_cores, order=df_volume['TIPO_PAGAMENTO'].value_counts().index)
            ax1.set_title('Volume por Tipo (Apenas Nota Mensal)')
            ax1.set_xlabel('')
            ax1.set_ylabel('Quantidade')
            ax1.tick_params(axis='x', rotation=45)

            if not df_volume.empty:
                sns.countplot(data=df_volume, y='Tab', ax=ax2, palette=paleta_cores, order=df_volume['Tab'].value_counts().index)
            ax2.set_title('Volume por Tabulação (Apenas Nota Mensal)')
            ax2.set_xlabel('Quantidade')
            ax2.set_ylabel('')

            ax_table = fig.add_subplot(gs[1, :])
            ax_table.axis('off')

            df_detalhe = df_bloco.groupby('Tab').agg(VALOR_PAGO=('VALOR_PAGO', 'sum'), NOTAS=('Conta_Volume', 'sum')).reset_index().sort_values('VALOR_PAGO', ascending=False)

            table_data = []
            for _, row in df_detalhe.iterrows():
                valor = row['VALOR_PAGO']
                notas = row['NOTAS']
                tm = valor / notas if notas > 0 else 0
                perc = (valor / valor_arrecadado) * 100 if valor_arrecadado > 0 else 0

                table_data.append([
                    row['Tab'], f"{notas}",
                    f"R$ {tm:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                    f"{perc:.1f}%",
                    f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                ])

            tm_total = valor_arrecadado / total_pagamentos if total_pagamentos > 0 else 0
            table_data.append([
                "TOTAL DO CONJUNTO", f"{total_pagamentos}",
                f"R$ {tm_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                "100.0%",
                f"R$ {valor_arrecadado:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            ])

            col_labels = ['Tabulação', 'Qtd Notas Mensais', 'Ticket Médio', 'Representação', 'Valor Arrecadado']
            table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 1.8)

            for (row, col), cell in table.get_celld().items():
                if col == 0: cell.set_text_props(ha='left')
                if row == 0: 
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#f2f2f2')
                elif row == len(table_data): 
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#e6e6e6')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig) 

        # Renderizando os blocos na tela
        st.header("📈 Análise por Conjuntos")
        analisar_bloco(df_positivas, "Tabulações Positivas", "Greens_r")
        st.divider()
        analisar_bloco(df_negativas, "Tabulações Negativas", "Reds_r")
        st.divider()
        analisar_bloco(df_outras, "Outras Tabulações", "Blues_r")

        # =====================================================================
        # TABELA DE RESUMO CONSOLIDADA
        # =====================================================================
        st.header("📋 Resumo Financeiro Consolidado")

        fig_resumo, ax_resumo = plt.subplots(figsize=(14, 12)) 
        ax_resumo.axis('off') 

        ax_resumo.set_title("RESUMO FINANCEIRO POR CONJUNTO DE TABULAÇÃO\n"
                  f"(Considerando pagamentos realizados em até {limite_dias_pagamento} dias após a ligação)", 
                  fontsize=16, fontweight='bold', pad=20)

        categorias = [('Positivas', '#1e7122', '#e6f2e6'), ('Negativas', '#a81c1c', '#f9e6e6'), ('Outras', '#155380', '#e6eff9')]
        table_data = []
        cell_colors = []
        text_weights = []
        text_colors = []
        col_labels = ['Tabulação', 'Qtd Notas Mensais', 'Ticket Médio', 'Representação', 'Valor Arrecadado']

        for categoria, cor_texto, cor_fundo in categorias:
            df_cat = df_cruzado[df_cruzado['Categoria'] == categoria]
            total_conjunto = df_cat['VALOR_PAGO'].sum()
            notas_conjunto = df_cat['Conta_Volume'].sum()
            tm_conjunto = total_conjunto / notas_conjunto if notas_conjunto > 0 else 0

            total_fmt = f"R$ {total_conjunto:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            tm_fmt = f"R$ {tm_conjunto:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

            table_data.append([f"CONJUNTO: {categoria.upper()}", f"{notas_conjunto}", tm_fmt, "100.0%", total_fmt])
            cell_colors.append([cor_fundo] * 5)
            text_weights.append(['bold'] * 5)
            text_colors.append([cor_texto] * 5)

            if df_cat.empty:
                table_data.append(["Nenhuma tabulação encontrada.", "-", "-", "-", "-"])
                cell_colors.append(['white'] * 5)
                text_weights.append(['normal'] * 5)
                text_colors.append(['black'] * 5)
            else:
                df_detalhe = df_cat.groupby('Tab').agg(VALOR_PAGO=('VALOR_PAGO', 'sum'), NOTAS=('Conta_Volume', 'sum')).reset_index().sort_values('VALOR_PAGO', ascending=False)
                for _, row in df_detalhe.iterrows():
                    valor = row['VALOR_PAGO']
                    notas = row['NOTAS']
                    tm = valor / notas if notas > 0 else 0
                    perc = (valor / total_conjunto) * 100 if total_conjunto > 0 else 0

                    valor_fmt = f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    tm_fmt = f"R$ {tm:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    perc_fmt = f"{perc:.1f}%"

                    table_data.append([row['Tab'], f"{notas}", tm_fmt, perc_fmt, valor_fmt])
                    cell_colors.append(['white'] * 5)
                    text_weights.append(['normal'] * 5)
                    text_colors.append(['black'] * 5)

        total_geral_valor = df_cruzado['VALOR_PAGO'].sum()
        total_geral_notas = df_cruzado['Conta_Volume'].sum()
        tm_geral = total_geral_valor / total_geral_notas if total_geral_notas > 0 else 0

        total_geral_fmt = f"R$ {total_geral_valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        tm_geral_fmt = f"R$ {tm_geral:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        table_data.append(["TOTAL GERAL DA CAMPANHA", f"{total_geral_notas}", tm_geral_fmt, "100.0%", total_geral_fmt])
        cell_colors.append(['#d9d9d9'] * 5) 
        text_weights.append(['bold'] * 5)
        text_colors.append(['black'] * 5)

        table = ax_resumo.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)

        for (row, col), cell in table.get_celld().items():
            if row == 0: 
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor('#f2f2f2')
            else:
                data_row = row - 1
                cell.set_facecolor(cell_colors[data_row][col])
                cell.set_text_props(weight=text_weights[data_row][col], color=text_colors[data_row][col])

            if col == 0: cell.set_text_props(ha='left')
            elif col == 4: cell.set_text_props(ha='right')
            cell.set_edgecolor('#cccccc')

        table.auto_set_column_width(col=[0, 1, 2, 3, 4])
        plt.tight_layout()

        st.pyplot(fig_resumo)
        plt.close(fig_resumo)

        # =====================================================================
        # EXPORTAÇÃO PARA EXCEL
        # =====================================================================
        st.header("📥 Exportar Dados")

        df_consolidado_excel = df_cruzado.groupby(['Categoria', 'Tab']).agg(Volume_Notas_Mensais=('Conta_Volume', 'sum'), Valor_Total=('VALOR_PAGO', 'sum')).reset_index()

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_consolidado_excel.to_excel(writer, sheet_name='Resumo_Consolidado', index=False)
            df_contagem_contatos.to_excel(writer, sheet_name='Contatos_Por_Cliente', index=False)
            df_positivas.to_excel(writer, sheet_name='Base_Positivas', index=False)
            df_negativas.to_excel(writer, sheet_name='Base_Negativas', index=False)
            df_outras.to_excel(writer, sheet_name='Base_Outras', index=False)

        st.download_button(
            label="Baixar Relatório Excel Completo",
            data=output.getvalue(),
            file_name="Relatorio_Cruzamento_Tabulacoes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
