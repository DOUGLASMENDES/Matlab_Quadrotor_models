# Fazer a instalação da python engine encontrada na pasta
# /C/Program Files/MATLAB/R202xx/extern/engines/python/
# mais informações: https://www.mathworks.com/help/releases/R2024b/matlab/matlab_external/install-the-matlab-engine-for-python.html
python -m pip install '/C/Program Files/MATLAB/R2024b/extern/engines/python/'

#caso não funcione:
#python -m pip install matlabengine

#última alternativa:
#python '/C/Program Files/MATLAB/R2024b/extern/engines/python/setup.py' install


