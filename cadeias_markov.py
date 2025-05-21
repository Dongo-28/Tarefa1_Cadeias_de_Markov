import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

# ========== QUESTÃO 1 ==========
def matriz_transicao(n):
    tm = np.zeros((n, n))
    np.fill_diagonal(tm[1:, :-1], 0.8)
    tm[0, :-1] = 0.2
    tm[-1, -1] = 1.0
    return tm

# ========== QUESTÃO 2 ==========
def propagar(p0, k, tm):
    return np.linalg.matrix_power(tm, k) @ p0

# ========== QUESTÃO 3 ==========
def plotar_distribuicoes_probabilidade():
    n = 10
    tm = matriz_transicao(n)
    p0 = np.zeros(n)
    p0[0] = 1
    
    plt.figure(figsize=(10, 6))
    for passo in range(1, 11):
        pk = propagar(p0, passo, tm)
        plt.plot(range(n), pk, label=f'Passo {passo}')
    
    plt.xlabel('Estado')
    plt.ylabel('Probabilidade')
    plt.title('Distribuição de Probabilidade por Estado nos Primeiros 10 Passos')
    plt.legend()
    plt.grid(True)
    plt.savefig("qsn3.png")
    plt.close()

# ========== QUESTÃO 4 ==========
def num_passos(n):
    tm = matriz_transicao(n)
    p0 = np.zeros(n)
    p0[0] = 1
    passos = 0
    pk = p0.copy()
    while pk[-1] < 0.5:
        pk = tm @ pk
        passos += 1
    return passos

def plotar_passos_vs_n():
    valores_n = range(10, 41)
    passos = [num_passos(n) for n in valores_n]
    
    # Gráfico normal
    plt.figure(figsize=(10, 5))
    plt.plot(valores_n, passos)
    plt.xlabel('Número de Estados (n)')
    plt.ylabel('Passos para Alcançar 50% de Probabilidade')
    plt.title('Passos Necessários para Alcançar 50% de Probabilidade no Estado Final')
    plt.grid(True)
    plt.savefig("qsn4c.png")
    plt.close()
    
    # Gráfico semilog
    plt.figure(figsize=(10, 5))
    plt.semilogy(valores_n, passos)
    plt.xlabel('Número de Estados (n)')
    plt.ylabel('Passos para Alcançar 50% de Probabilidade (escala log)')
    plt.title('Gráfico Semilog dos Passos Necessários para Alcançar 50% de Probabilidade')
    plt.grid(True)
    plt.savefig("qsn4c_semilogy.png")
    plt.close()

# ========== QUESTÃO 5 ==========
def amostrar(tm, k, s0):
    estados = [s0]
    estado_atual = s0
    for _ in range(k):
        probs = tm[:, estado_atual]
        estado_atual = rnd.choice(len(probs), p=probs)
        estados.append(estado_atual)
    return estados

def plotar_amostras():
    tm = matriz_transicao(10)
    k = 100
    s0 = 0
    plt.figure(figsize=(10, 6))
    for i in range(20):
        estados = amostrar(tm, k, s0)
        plt.plot(estados)
    plt.xlabel('Passo')
    plt.ylabel('Estado')
    plt.title('20 Trajetórias Amostradas da Cadeia de Markov')
    plt.grid(True)
    plt.savefig("qsn5.png")
    plt.close()

# ========== QUESTÃO 6 ==========
def plotar_estado_medio():
    n = 25
    tm = matriz_transicao(n)
    k = 100
    s0 = 0
    num_amostras = 1000
    todas_amostras = np.zeros((num_amostras, k+1))
    for i in range(num_amostras):
        todas_amostras[i] = amostrar(tm, k, s0)
    estados_medios = np.mean(todas_amostras, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(estados_medios)
    plt.xlabel('Passo de Tempo')
    plt.ylabel('Estado Médio')
    plt.title('Progressão do Estado Médio ao Longo de 100 Passos (n=25)')
    plt.grid(True)
    plt.savefig("qsn6.png")
    plt.close()

# ========== QUESTÃO 7 ==========
def verificar_amostragem():
    n = 25
    tm = matriz_transicao(n)
    k = 100
    s0 = 0
    num_amostras = 1000
    estados_finais = []
    for _ in range(num_amostras):
        estados = amostrar(tm, k, s0)
        estados_finais.append(estados[-1])
    p0 = np.zeros(n)
    p0[s0] = 1
    pk = propagar(p0, k, tm)
    contagens_esperadas = pk * num_amostras
    plt.figure(figsize=(10, 6))
    plt.hist(estados_finais, bins=np.arange(-0.5, n, 1), alpha=0.7, label='Amostrado')
    plt.plot(range(n), contagens_esperadas, 'r-', marker='o', label='Teórico')
    plt.xlabel('Estado Final')
    plt.ylabel('Contagem')
    plt.title('Comparação entre Distribuições de Estados Finais Amostrados e Teóricos')
    plt.legend()
    plt.grid(True)
    plt.savefig("qsn7.png")
    plt.close()

# ========== EXECUTAR TUDO ==========
if __name__ == "__main__":
    plotar_distribuicoes_probabilidade()  # Q3
    plotar_passos_vs_n()                  # Q4
    plotar_amostras()                     # Q5
    plotar_estado_medio()                 # Q6
    verificar_amostragem()                # Q7
    print("Todos os gráficos foram gerados com sucesso!")