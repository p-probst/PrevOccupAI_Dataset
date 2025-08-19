#!/bin/bash

# Script 3: Criar ambiente virtual e instalar dependências
echo "=== Criação do Ambiente Virtual ==="
echo "Data/Hora: $(date)"

cd ~/projects/PrevOccupAI_Dataset

# Criar ambiente virtual
echo "1. Criando ambiente virtual..."
python3 -m venv prevOccupAI_venv
if [ $? -ne 0 ]; then
    echo "ERRO: Falha na criação do ambiente virtual!"
    exit 1
fi

# Ativar ambiente virtual
echo "2. Ativando ambiente virtual..."
source prevOccupAI_venv/bin/activate

# Atualizar pip
echo "3. Atualizando pip..."
pip install --upgrade pip

# Instalar dependências básicas
echo "4. Instalando dependências essenciais..."
pip install numpy pandas scikit-learn joblib matplotlib seaborn tqdm

# Instalar dependências do requirements.txt se existir
if [ -f "requirements.txt" ]; then
    echo "5. Instalando dependências do requirements.txt..."
    pip install -r requirements.txt
else
    echo "5. requirements.txt não encontrado, instalando dependências adicionais..."
    pip install tsfel ipython jupyter
fi

# Verificar instalações
echo "6. Verificando instalações..."
python -c "import numpy, pandas, sklearn, joblib, matplotlib, seaborn; print('Todas as dependências instaladas com sucesso!')"

if [ $? -eq 0 ]; then
    echo "=== Ambiente virtual configurado com sucesso ==="
    echo "Próximo passo: Executar 04_run_model_selection.sh"
else
    echo "ERRO: Problema com as dependências!"
    exit 1
fi
