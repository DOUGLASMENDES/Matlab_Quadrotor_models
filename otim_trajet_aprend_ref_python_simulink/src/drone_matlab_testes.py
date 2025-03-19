import matlab.engine
import time

nome_modelo_simulink = 'modelo_naolinear'

# Função para calcular o erro em relação à posição desejada
def calculate_error(current_z, desired_z):
    return abs(current_z - desired_z) / desired_z


# Inicializa a MATLAB Engine
print('Inicializando matlab...')
#eng = matlab.engine.connect_matlab()
eng = matlab.engine.start_matlab()
print('Matlab iniciado.')

# Carrega o modelo Simulink
print('Carregando modelo simulink...')
#eng.load_system(nome_modelo_simulink) 
eng.eval(f"load_system('{nome_modelo_simulink}')", nargout=0)
print('Modelo simulink iniciado.') 
eng.desktop(nargout=0)


# Lista de posições desejadas (z_d)
z_d_list = [1.5, 4.0, 2.0, 3.5, 5.0, 6.0]  # Substitua pelos valores desejados

# Inicia a simulação
#eng.set_param(nome_modelo_simulink, 'SimulationCommand', 'start')
eng.eval(f"set_param('{nome_modelo_simulink}', 'SimulationCommand', 'start')", nargout=0)
print('Simulação iniciada...')

# Loop para enviar as posições desejadas ao Simulink
for z_d in z_d_list:
    print(f"Enviando nova posição desejada: {z_d}")    
    # Atualiza a posição desejada no MATLAB workspace
    #eng.set_variable('vzd3', z_d)
    #eng.eval(f'vz_desejada = {z_d};',nargout=0)
    eng.workspace['vz_desejada'] = z_d  # 'vzd3' deve ser usado no modelo Simulink

    # Aguarda a convergência
    while True:
        # Obtem a posição atual e o erro do Simulink
        #current_z = eng.get_variable('z')  # Variável que contém a posição atual
        current_z = eng.workspace['z_atual']  # Supondo que 'z' seja a variável de posição atual
        error = calculate_error(current_z, z_d)

        print(f"Posição atual: {current_z}, Erro: {error * 100:.2f}%")

        if error < 0.05:  # Erro menor que 5%
            print(f"Erro aceitável alcançado para z_d = {z_d}")
            break
        
        time.sleep(0.1)  # Aguarda 0.1s antes de verificar novamente


# Para a simulação
eng.eval(f"set_param('{nome_modelo_simulink}', 'SimulationCommand', 'stop')", nargout=0)
#eng.set_param(nome_modelo_simulink, 'SimulationCommand', 'stop')

# Fecha o MATLAB Engine
eng.quit()

print("Todas as posições desejadas foram enviadas com sucesso!")
