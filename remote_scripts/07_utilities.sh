#!/bin/bash

# Script 7: Utilitários para monitorização e transferência
echo "=== Utilitários de Monitorização ==="
echo "Data/Hora: $(date)"

cd ~/projects/PrevOccupAI_Dataset

echo "Escolha uma operação:"
echo "1) Verificar uso de recursos (CPU, Memória, Disco)"
echo "2) Verificar processos Python"
echo "3) Ver logs disponíveis"
echo "4) Verificar resultados gerados"
echo "5) Gerar comandos para transferência de ficheiros"
echo "6) Limpar logs antigos"
echo "7) Parar todos os processos Python"

read -p "Opção (1-7): " choice

case $choice in
    1)
        echo "=== USO DE RECURSOS ==="
        echo "CPU e Memória:"
        top -b -n1 | head -15
        echo ""
        echo "Uso de disco:"
        df -h
        echo ""
        echo "Memória detalhada:"
        free -h
        ;;
    2)
        echo "=== PROCESSOS PYTHON ==="
        ps aux | grep python | grep -v grep
        echo ""
        echo "PIDs dos processos:"
        pgrep python
        ;;
    3)
        echo "=== LOGS DISPONÍVEIS ==="
        if [ -d "logs" ]; then
            ls -la logs/
            echo ""
            echo "Para ver um log em tempo real:"
            echo "tail -f logs/[nome_do_ficheiro].log"
        else
            echo "Nenhum diretório de logs encontrado"
        fi
        ;;
    4)
        echo "=== RESULTADOS GERADOS ==="
        echo "Estrutura de resultados:"
        find . -name "Results" -type d 2>/dev/null
        find . -name "production_models" -type d 2>/dev/null
        echo ""
        echo "Ficheiros .csv:"
        find . -name "*.csv" -type f 2>/dev/null
        echo ""
        echo "Modelos .joblib:"
        find . -name "*.joblib" -type f 2>/dev/null
        ;;
    5)
        echo "=== COMANDOS PARA TRANSFERÊNCIA ==="
        echo "Execute estes comandos no teu Mac (terminal local):"
        echo ""
        echo "# Transferir resultados:"
        echo "scp -r lab@100.94.245.106:~/projects/PrevOccupAI_Dataset/Results ~/Documents/projects/PrevOccupAI_Dataset/"
        echo ""
        echo "# Transferir modelos:"
        echo "scp -r lab@100.94.245.106:~/projects/PrevOccupAI_Dataset/HAR/production_models ~/Documents/projects/PrevOccupAI_Dataset/HAR/"
        echo ""
        echo "# Transferir logs:"
        echo "scp -r lab@100.94.245.106:~/projects/PrevOccupAI_Dataset/logs ~/Documents/projects/PrevOccupAI_Dataset/"
        echo ""
        echo "# Transferir projeto completo:"
        echo "scp -r lab@100.94.245.106:~/projects/PrevOccupAI_Dataset ~/Documents/projects/"
        ;;
    6)
        echo "=== LIMPEZA DE LOGS ==="
        if [ -d "logs" ]; then
            echo "Logs atuais:"
            ls -la logs/
            read -p "Apagar todos os logs? (y/N): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                rm -f logs/*
                echo "Logs apagados!"
            else
                echo "Operação cancelada"
            fi
        else
            echo "Nenhum diretório de logs encontrado"
        fi
        ;;
    7)
        echo "=== PARAR PROCESSOS PYTHON ==="
        echo "Processos Python encontrados:"
        ps aux | grep python | grep -v grep
        echo ""
        read -p "Parar TODOS os processos Python? (y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            pkill python
            echo "Processos Python interrompidos!"
        else
            echo "Operação cancelada"
        fi
        ;;
    *)
        echo "Opção inválida!"
        ;;
esac
