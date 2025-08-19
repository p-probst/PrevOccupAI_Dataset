#!/bin/bash

# Script 6: Executar scripts em background com monitorização
echo "=== Execução em Background com Logs ==="
echo "Data/Hora: $(date)"

cd ~/projects/PrevOccupAI_Dataset

# Ativar ambiente virtual
source prevOccupAI_venv/bin/activate

# Criar diretório para logs
mkdir -p logs

echo "Escolha a operação a executar:"
echo "1) Seleção de Modelos (pode demorar várias horas)"
echo "2) Treino do Modelo de Produção"
echo "3) Ambos (sequencialmente)"
echo "4) Verificar processos em execução"
echo "5) Ver logs em tempo real"

read -p "Opção (1-5): " choice

case $choice in
    1)
        echo "Executando seleção de modelos em background..."
        nohup python remote_scripts/04_run_model_selection.py > logs/model_selection.log 2>&1 &
        echo "PID: $!"
        echo "Log: logs/model_selection.log"
        echo "Para monitorizar: tail -f logs/model_selection.log"
        ;;
    2)
        echo "Executando treino do modelo de produção em background..."
        nohup python remote_scripts/05_run_production_training.py > logs/production_training.log 2>&1 &
        echo "PID: $!"
        echo "Log: logs/production_training.log"
        echo "Para monitorizar: tail -f logs/production_training.log"
        ;;
    3)
        echo "Executando ambos sequencialmente em background..."
        nohup bash -c "
        echo 'Iniciando seleção de modelos...'
        python remote_scripts/04_run_model_selection.py
        echo 'Seleção concluída. Iniciando treino de produção...'
        python remote_scripts/05_run_production_training.py
        echo 'Processo completo finalizado!'
        " > logs/full_pipeline.log 2>&1 &
        echo "PID: $!"
        echo "Log: logs/full_pipeline.log"
        echo "Para monitorizar: tail -f logs/full_pipeline.log"
        ;;
    4)
        echo "Processos Python em execução:"
        ps aux | grep python | grep -v grep
        ;;
    5)
        echo "Logs disponíveis:"
        ls -la logs/ 2>/dev/null || echo "Nenhum log encontrado"
        echo ""
        echo "Para ver log em tempo real, usa:"
        echo "tail -f logs/[nome_do_log].log"
        ;;
    *)
        echo "Opção inválida!"
        ;;
esac
