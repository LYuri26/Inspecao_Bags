### 🌳 Estrutura de Arquivos por Fase

#### 📌 Fase 1: Núcleo Básico (Semana 1)

```
projeto/
├── .env                # Variáveis de ambiente
├── requirements.txt    # Dependências Python
└── src/
    ├── main.py         # Ponto de entrada CLI
    └── detector/       # Módulo de detecção
        ├── __init__.py
        ├── model.py    # Carregador do modelo YOLO
        └── processor.py # Processamento de imagem
```

**Tarefas:**

1. Criar ambiente virtual e instalar PyTorch+OpenCV
2. Implementar carregamento do modelo YOLO em `model.py`
3. Desenvolver pipeline de processamento em `processor.py`
4. Testar com imagens locais via CLI no `main.py`

#### 📌 Fase 2: Camada de Inteligência (Semana 2)

```
src/
├── ...
└── core/
    ├── classifier.py   # Classificação de gravidade
    └── validator.py    # Validação por políticas
```

**Tarefas:**

1. Implementar níveis 1-3 de gravidade em `classifier.py`
2. Criar motor de regras empresariais em `validator.py`
3. Conectar módulos de detecção e classificação
4. Adicionar testes para cada tipo de defeito

#### 📌 Fase 3: Interface (Semanas 3-4)

```
src/
├── ...
└── ui/
    ├── main_window/    # Interface principal
    │   ├── window.py   # Classe da janela
    │   └── layout.py   # Configuração do layout
    └── widgets/
        ├── camera.py   # Visualização da câmera
        └── alerts.py   # Sistema de alertas
```

**Tarefas:**

1. Instalar PyQt5 e estruturar janela principal
2. Desenvolver widget de câmera com OpenCV
3. Criar alertas visuais por gravidade (cores)
4. Implementar lista interativa de defeitos

#### 📌 Fase 4: Dados (Semana 5)

```
src/
├── ...
└── data/
    ├── models/         # Modelos de dados
    │   ├── company.py  # Perfis de empresas
    │   └── defect.py   # Registros de defeitos
    ├── crud/          # Operações CRUD
    │   └── manager.py  # Gerenciamento do banco
    └── migrations/    # Scripts SQL
        └── v1_init.sql # Schema inicial
```

**Tarefas:**

1. Modelar estrutura MySQL para empresas/inspeções
2. Implementar operações CRUD em `manager.py`
3. Conectar interface ao banco de dados
4. Configurar geração automática de relatórios PDF

### 🔄 Alinhamento com o README

1. **Detecção de Defeitos**

   - Arquivo: `detector/processor.py`
   - Implementa: Rasgos, Cortes, Sujeiras, Manchas, Descosturas
   - Saída: Coordenadas + confiança

2. **Gravidade dos Defeitos**

   - Arquivo: `core/classifier.py`
   - Sistema: Níveis 1-3 (leve, moderado, grave)
   - Critérios: Tipo + área afetada

3. **Políticas Empresariais**

   - Arquivo: `core/validator.py`
   - Funcionalidade: Limiares personalizados por empresa
   - Exemplo: Aceitar manchas mas não cortes

4. **Relatórios**
   - Arquivos:
     - `ui/widgets/reports.py` (gera PDF)
     - `data/crud/manager.py` (consulta dados)

### 📅 Cronograma Detalhado

| Semana | Foco         | Arquivos Principais | Entregável                          |
| ------ | ------------ | ------------------- | ----------------------------------- |
| 1      | Detecção     | detector/\*.py      | CLI que detecta 5 tipos de defeitos |
| 2      | Inteligência | core/\*.py          | Classificação por gravidade         |
| 3-4    | Interface    | ui/\*_/_.py         | Sistema completo de inspeção visual |
| 5      | Dados        | data/\*_/_.py       | Integração MySQL + relatórios       |
