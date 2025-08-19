#!/usr/bin/env python3
"""
Script para treinar o modelo de produção HAR
Configurado para:
- Balanceamento: main_classes
- Window size: 5000 samples (5 segundos)
- Normalization: standard (melhor performance geralmente)
- Features: 25 (valor padrão, ajustar baseado nos resultados da seleção)
"""

import os
import sys
import time
from datetime import datetime

# Adicionar o diretório do projeto ao Python path
project_root = os.path.expanduser("~/projects/PrevOccupAI_Dataset")
sys.path.insert(0, project_root)

def main():
    print("=" * 60)
    print("TREINO DO MODELO DE PRODUÇÃO HAR - PrevOccupAI")
    print("=" * 60)
    print(f"Data/Hora de início: {datetime.now()}")
    print(f"Diretório do projeto: {project_root}")
    
    try:
        # Importar as funções necessárias
        from HAR.model_selection import train_production_model
        print("✓ Módulos HAR importados com sucesso")
        
        # Configurar parâmetros
        # AJUSTAR ESTE CAMINHO PARA OS TEUS DADOS EXTRAÍDOS
        data_path = os.path.join(project_root, "extracted_features")  # AJUSTAR!
        num_features_retain = 25  # Ajustar baseado nos resultados da seleção
        balancing_type = "main_classes"
        norm_type = "standard"  # Normalmente o melhor
        window_size_samples = 5000  # 5 segundos a 1000Hz
        
        print(f"\nParâmetros configurados:")
        print(f"  - Data path: {data_path}")
        print(f"  - Número de features: {num_features_retain}")
        print(f"  - Balancing type: {balancing_type}")
        print(f"  - Normalization type: {norm_type}")
        print(f"  - Window size: {window_size_samples} samples")
        
        # Verificar se o caminho dos dados existe
        if not os.path.exists(data_path):
            print(f"\n❌ ERRO: Caminho dos dados não encontrado: {data_path}")
            print("\nPor favor, ajusta a variável 'data_path' no script para o caminho correto dos teus dados extraídos.")
            return 1
        
        # Verificar se a pasta específica de normalização existe
        norm_folder = f"w_{window_size_samples}_sc_{norm_type}"
        norm_path = os.path.join(data_path, norm_folder)
        
        if not os.path.exists(norm_path):
            print(f"\n❌ ERRO: Pasta de normalização não encontrada: {norm_path}")
            print(f"\nPastas disponíveis em {data_path}:")
            for item in os.listdir(data_path):
                if os.path.isdir(os.path.join(data_path, item)):
                    print(f"  - {item}")
            return 1
        
        print(f"\n✓ Pasta de dados encontrada: {norm_path}")
        
        # Verificar resultados da seleção anterior (opcional)
        results_path = os.path.join(project_root, "Results")
        if os.path.exists(results_path):
            print(f"\n📊 Resultados da seleção anteriores encontrados em: {results_path}")
            print("Recomenda-se analisar os resultados para otimizar os parâmetros.")
        
        # Confirmar antes de prosseguir
        print(f"\n" + "="*60)
        print("PRONTO PARA INICIAR O TREINO DO MODELO DE PRODUÇÃO")
        print("Este processo pode demorar algumas horas...")
        print("="*60)
        
        # Executar treino do modelo de produção
        start_time = time.time()
        print(f"\n🚀 Iniciando treino do modelo de produção... {datetime.now()}")
        
        train_production_model(data_path, num_features_retain, balancing_type, norm_type, window_size_samples)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ TREINO DO MODELO DE PRODUÇÃO CONCLUÍDO!")
        print(f"Tempo total: {duration/3600:.2f} horas ({duration/60:.1f} minutos)")
        print(f"Data/Hora de conclusão: {datetime.now()}")
        
        # Verificar modelo produzido
        model_path = os.path.join(project_root, "HAR", "production_models", f"{window_size_samples}_w_size")
        if os.path.exists(model_path):
            print(f"\n🎯 Modelo de produção salvo em: {model_path}")
            
            # Listar ficheiros do modelo
            for item in os.listdir(model_path):
                print(f"  - {item}")
        
        print(f"\n✨ Processo completo! O modelo está pronto para ser usado.")
        
        # Instruções para transferir resultados
        print(f"\n📁 Para transferir os resultados para o teu Mac, executa no teu terminal local:")
        print(f"scp -r lab@100.94.245.106:~/projects/PrevOccupAI_Dataset/Results ~/Documents/projects/PrevOccupAI_Dataset/")
        print(f"scp -r lab@100.94.245.106:~/projects/PrevOccupAI_Dataset/HAR/production_models ~/Documents/projects/PrevOccupAI_Dataset/HAR/")
        
        return 0
        
    except ImportError as e:
        print(f"\n❌ ERRO: Problema ao importar módulos: {e}")
        print("Verifica se o ambiente virtual está ativo e as dependências instaladas.")
        return 1
    except Exception as e:
        print(f"\n❌ ERRO durante a execução: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
