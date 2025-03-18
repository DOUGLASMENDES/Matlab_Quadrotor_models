import re
import pandas as pd
from datetime import datetime

# Inicializa listas para armazenar os dados
rodadas = []
tempos_execucao = []
epsilons = []
custos = []

# Define o padrão regex para extrair os campos do log
padrao = re.compile(
    r'RODADA: (?P<rodada>\d+);hora: (?P<hora>[\d\-:\. ]+);Rodada: \d+;Epsilon: (?P<epsilon>[\d\.]+);Custo: (?P<custo>\d+)'
)

# Lê o arquivo de log
with open('log_treinamento.txt', 'r') as arquivo:
    linhas = arquivo.readlines()

# Processa cada linha do log
hora_anterior = None
for linha in linhas:
    correspondencia = padrao.match(linha)
    if correspondencia:
        dados = correspondencia.groupdict()
        rodada = int(dados['rodada'])
        hora = datetime.strptime(dados['hora'], '%Y-%m-%d %H:%M:%S.%f')
        epsilon = float(dados['epsilon'])
        custo = int(dados['custo'])

        rodadas.append(rodada)
        epsilons.append(epsilon)
        custos.append(custo)

        if hora_anterior:
            tempo_execucao = (hora - hora_anterior).total_seconds() / 60  # Converte para minutos
            tempos_execucao.append(tempo_execucao)
        else:
            tempos_execucao.append(0)  # Para a primeira rodada, o tempo de execução é zero

        hora_anterior = hora

# Cria um DataFrame com os dados extraídos
df = pd.DataFrame({
    'Rodada': rodadas,
    'Tempo de Execução (min)': tempos_execucao,
    'Epsilon': epsilons,
    'Custo': custos
})

# Remove a primeira linha, pois o tempo de execução é zero
df = df.iloc[1:].reset_index(drop=True)

print(df)

import matplotlib.pyplot as plt

# Configura o tamanho da figura
plt.figure(figsize=(10, 6))

# Plota o Tempo de Execução
#plt.plot(df['Rodada'], df['Tempo de Execução (min)'], label='Tempo de Execução (min)')

# Plota o Epsilon
#plt.plot(df['Rodada'], df['Epsilon'], label='Epsilon')

# Plota o Custo
plt.plot(df['Rodada'], df['Custo'], label='Custo')

# Configurações do gráfico
plt.xlabel('Episódio/Rodada')
plt.ylabel('Função de Custo')
plt.yscale('log')  # Define a escala logarítmica no eixo y
plt.title('Evolução da Função Custo por Episódio/Rodada')
plt.legend()
plt.grid(True, which="both", ls="--")

# Exibe o gráfico
plt.show()
