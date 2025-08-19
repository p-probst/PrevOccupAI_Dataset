#!/bin/bash

# Script 1: Configuração do ambiente no PC remoto
echo "=== Configuração do Ambiente HAR - PrevOccupAI ==="
echo "Data/Hora: $(date)"

# Criar diretório do projeto
echo "1. Criando diretório do projeto..."
mkdir -p ~/projects_GBarros
cd ~/projects_GBarros

# Verificar se Python3 está disponível
echo "2. Verificando Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERRO: Python3 não encontrado!"
    exit 1
fi

# Verificar se git está disponível
echo "3. Verificando Git..."
git --version
if [ $? -ne 0 ]; then
    echo "AVISO: Git não encontrado. Será necessário transferir ficheiros manualmente."
fi

# Verificar espaço em disco
echo "4. Verificando espaço em disco..."
df -h ~/

# Verificar memória
echo "5. Verificando memória..."
free -h

echo "6. Verificando CPU..."
nproc

# Verificar GPU
echo "7. Verificando GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detectada:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
else
    echo "AVISO: nvidia-smi não encontrado. GPU NVIDIA não detectada ou drivers não instalados."
fi

echo "=== Configuração inicial completa ==="
echo "Próximo passo: Executar 02_clone_project.sh ou transferir ficheiros manualmente"
