# Guia de Execução - HAR Training Pipeline
# PrevOccupAI Dataset - Execução Remota

## Configuração dos Parâmetros
- **Balanceamento**: main_classes
- **Window Size**: 5000 samples (5 segundos a 1000Hz)
- **Normalização padrão**: standard (para modelo de produção)

## Sequência de Execução no PC Remoto

### 1. Conectar ao PC Remoto
```bash
ssh lab@100.94.245.106
# Password: 12341
```

### 2. Configuração Inicial
```bash
# Executar configuração do ambiente
chmod +x remote_scripts/01_setup_environment.sh
./remote_scripts/01_setup_environment.sh
```

### 3. Clonar/Transferir Projeto
```bash
# Tentar clonagem automática
chmod +x remote_scripts/02_clone_project.sh
./remote_scripts/02_clone_project.sh

# SE A CLONAGEM FALHAR, executar no Mac (terminal local):
# scp -r /Users/goncalobarros/Documents/projects/PrevOccupAI_Dataset lab@100.94.245.106:~/projects/
```

### 4. Criar Ambiente Virtual
```bash
chmod +x remote_scripts/03_create_venv.sh
./remote_scripts/03_create_venv.sh
```

### 5. IMPORTANTE: Ajustar Caminho dos Dados
**ANTES de executar o treino, editar os scripts para o caminho correto dos dados:**

```bash
# Editar o caminho nos scripts Python
nano remote_scripts/04_run_model_selection.py
# Alterar linha: data_path = os.path.join(project_root, "extracted_features")
# Para: data_path = "/path/to/your/extracted/features"

nano remote_scripts/05_run_production_training.py
# Fazer a mesma alteração
```

### 6. Execução do Training Pipeline

#### Opção A: Execução Interativa
```bash
# Seleção de modelos
chmod +x remote_scripts/04_run_model_selection.py
python remote_scripts/04_run_model_selection.py

# Treino de produção (após seleção)
chmod +x remote_scripts/05_run_production_training.py
python remote_scripts/05_run_production_training.py
```

#### Opção B: Execução em Background (Recomendado)
```bash
chmod +x remote_scripts/06_run_background.sh
./remote_scripts/06_run_background.sh
# Escolher opção 3 para execução completa
```

### 7. Monitorização
```bash
# Verificar progresso
tail -f logs/full_pipeline.log

# Verificar recursos
chmod +x remote_scripts/07_utilities.sh
./remote_scripts/07_utilities.sh
```

### 8. Transferir Resultados (no Mac)
```bash
# Resultados
scp -r lab@100.94.245.106:~/projects/PrevOccupAI_Dataset/Results ~/Documents/projects/PrevOccupAI_Dataset/

# Modelos
scp -r lab@100.94.245.106:~/projects/PrevOccupAI_Dataset/HAR/production_models ~/Documents/projects/PrevOccupAI_Dataset/HAR/

# Logs
scp -r lab@100.94.245.106:~/projects/PrevOccupAI_Dataset/logs ~/Documents/projects/PrevOccupAI_Dataset/
```

## Estrutura de Ficheiros Esperada

### Input (Dados Extraídos):
```
extracted_features/
├── w_5000_sc_none/
├── w_5000_sc_minmax/
└── w_5000_sc_standard/
```

### Output (Resultados):
```
Results/
└── ML/
    └── 5000_w_size/
        └── num_classes_X/
            ├── Random Forest_f25_wNorm-standard.csv
            ├── KNN_f25_wNorm-standard.csv
            └── SVM_f25_wNorm-standard.csv

HAR/production_models/
└── 5000_w_size/
    ├── HAR_model_5000.joblib
    ├── cfg_file_production_model.json
    └── ConfusionMatrix_5000_w_size.png
```

## Comandos Úteis Durante Execução

```bash
# Verificar processos
ps aux | grep python

# Verificar uso de recursos
htop
# ou
top

# Verificar espaço em disco
df -h

# Parar processo específico
kill [PID]

# Verificar logs
ls -la logs/
tail -f logs/[nome_do_log].log
```

## Tempos Estimados
- **Seleção de modelos**: 3-8 horas (depende dos dados)
- **Treino de produção**: 30 minutos - 2 horas
- **Total**: 4-10 horas aproximadamente

## Troubleshooting

### Se houver erros de import:
```bash
source prevOccupAI_venv/bin/activate
pip install [missing_package]
```

### Se faltar memória:
- Usar menos features (editar num_features_retain)
- Processar um tipo de normalização de cada vez

### Se houver erro de caminho:
- Verificar se data_path está correto nos scripts Python
- Verificar se as pastas w_5000_sc_* existem
