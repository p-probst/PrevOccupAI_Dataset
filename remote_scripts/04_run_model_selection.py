#!/usr/bin/env python3
"""
Script para executar a seleção de modelos HAR
Configurado para:
- Balanceamento: main_classes
- Window size: 5000 samples (5 segundos)
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
    print("SELEÇÃO DE MODELOS HAR - PrevOccupAI")
    print("=" * 60)
    print(f"Data/Hora de início: {datetime.now()}")
    print(f"Diretório do projeto: {project_root}")
    
    try:
        # Importar as funções necessárias
        from HAR.model_selection import perform_model_selection
        print("✓ Módulos HAR importados com sucesso")
        
        # Configurar parâmetros
        # AJUSTAR ESTE CAMINHO PARA OS TEUS DADOS EXTRAÍDOS
        data_path = os.path.join(project_root, "extracted_features")  # AJUSTAR!
        balancing_type = "main_classes"
        window_size_samples = 5000  # 5 segundos a 1000Hz
        
        print(f"\nParâmetros configurados:")
        print(f"  - Data path: {data_path}")
        print(f"  - Balancing type: {balancing_type}")
        print(f"  - Window size: {window_size_samples} samples")
        
        # Verificar se o caminho dos dados existe
        if not os.path.exists(data_path):
            print(f"\n❌ ERRO: Caminho dos dados não encontrado: {data_path}")
            print("\nPor favor, ajusta a variável 'data_path' no script para o caminho correto dos teus dados extraídos.")
            print("Exemplo: data_path = '/path/to/your/extracted/features'")
            return 1
        
        print(f"\n✓ Caminho dos dados encontrado: {data_path}")
        
        # Verificar subpastas esperadas
        expected_folders = [
            f"w_{window_size_samples}_sc_none",
            f"w_{window_size_samples}_sc_minmax", 
            f"w_{window_size_samples}_sc_standard"
        ]
        
        print(f"\nVerificando subpastas esperadas:")
        for folder in expected_folders:
            folder_path = os.path.join(data_path, folder)
            if os.path.exists(folder_path):
                print(f"  ✓ {folder}")
            else:
                print(f"  ❌ {folder} - NÃO ENCONTRADA")
        
        # Confirmar antes de prosseguir
        print(f"\n" + "="*60)
        print("PRONTO PARA INICIAR A SELEÇÃO DE MODELOS")
        print("Este processo pode demorar várias horas...")
        print("="*60)
        
        # Executar seleção de modelos
        start_time = time.time()
        print(f"\n🚀 Iniciando seleção de modelos... {datetime.now()}")
        
        perform_model_selection(data_path, balancing_type, window_size_samples)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ SELEÇÃO DE MODELOS CONCLUÍDA!")
        print(f"Tempo total: {duration/3600:.2f} horas ({duration/60:.1f} minutos)")
        print(f"Data/Hora de conclusão: {datetime.now()}")
        
        # Verificar resultados
        results_path = os.path.join(project_root, "Results")
        if os.path.exists(results_path):
            print(f"\n📊 Resultados salvos em: {results_path}")
            
            # Listar ficheiros de resultados
            for root, dirs, files in os.walk(results_path):
                for file in files:
                    if file.endswith('.csv'):
                        print(f"  - {os.path.join(root, file)}")
        
        print(f"\n✨ Próximo passo: Executar 05_run_production_training.py")
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
