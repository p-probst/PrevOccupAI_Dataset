#!/bin/bash

# Script 2: Clonar/Transferir projeto
echo "=== Clonagem/Transferência do Projeto ==="
echo "Data/Hora: $(date)"

cd ~/projects

# Opção A: Clonar via Git (se disponível)
echo "Tentando clonar repositório via Git..."
git clone https://github.com/eLbARROS13/PrevOccupAI_Dataset.git
if [ $? -eq 0 ]; then
    echo "Repositório clonado com sucesso!"
    cd PrevOccupAI_Dataset
else
    echo "AVISO: Falha na clonagem via Git."
    echo "Execute no teu Mac (numa janela de terminal separada):"
    echo "scp -r /Users/goncalobarros/Documents/projects/PrevOccupAI_Dataset lab@100.94.245.106:~/projects/"
    echo ""
    echo "Aguardando transferência manual..."
    echo "Pressiona ENTER quando a transferência estiver completa..."
    read -p ""
    
    # Verificar se o projeto foi transferido
    if [ -d "PrevOccupAI_Dataset" ]; then
        echo "Projeto encontrado! Continuando..."
        cd PrevOccupAI_Dataset
    else
        echo "ERRO: Projeto não encontrado!"
        exit 1
    fi
fi

# Verificar estrutura do projeto
echo "Verificando estrutura do projeto..."
ls -la

# Verificar se as pastas essenciais existem
if [ ! -d "HAR" ]; then
    echo "ERRO: Pasta HAR não encontrada!"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "AVISO: requirements.txt não encontrado!"
fi

echo "=== Projeto configurado com sucesso ==="
echo "Próximo passo: Executar 03_create_venv.sh"
