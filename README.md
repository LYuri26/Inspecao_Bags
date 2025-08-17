# Sistema de Inspeção Automatizada de Sacolas Plásticas

### 1. Objetivo

Desenvolvimento sob demanda de um sistema **industrial de inspeção visual automatizada** para sacolas plásticas, com foco em **detecção de defeitos** em linha de produção. Software personalizado, entregue com exclusividade ao contratante, que deterá a propriedade integral do produto final. O desenvolvedor fornecerá entrega técnica e suporte por período definido, sem retenção de direitos sobre o sistema entregue.

---

### 2. Características do Sistema

- **Visão Computacional Avançada**: Identificação automática dos principais defeitos visuais em sacolas plásticas, garantindo controle rigoroso de qualidade.
- **Interface Gráfica Industrial**: Aplicação intuitiva, com navegação funcional, desenvolvida para uso em ambiente fabril.
- **Processamento Local**: Imagens capturadas por câmera são processadas diretamente na máquina, sem necessidade de conexão externa.
- **Armazenamento Organizado e Seguro**: Dados de inspeções armazenados localmente em estrutura hierarquizada por empresa e inspeção, com backup automático para garantir integridade.
- **Cadastro e Gestão de Empresas**: Controle detalhado das políticas de aceitação e histórico de inspeções, garantindo alinhamento com requisitos do cliente.
- **Geração de Registros**: Produção automática de arquivos detalhados (ex: `.txt`) por inspeção para auditoria e conformidade.
- **Escalabilidade e Integração**: Possibilidade futura de integração com sistemas internos da empresa, mediante nova negociação.

---

### 3. Estrutura do Projeto

Organização modular que separa claramente as responsabilidades do sistema, facilitando manutenção, evolução e customização:

```
Inspecao_Bags/
├── .venv/                         # Ambiente virtual Python
├── cadastros/                     # Dados de empresas e inspeções
│   └── Lenon Yuri/                # Pasta de exemplo para empresa
│       └── Lenon Yuri.json        # Dados da empresa em JSON
├── dataset_sacolas/               # Dataset de imagens para treino
│   ├── baixadas/                  # Imagens capturadas brutas
│   ├── images/                    # Imagens organizadas para treino
│   │   ├── train/                 # Imagens treino
│   │   │   ├── aug_0_img_0044.jpg
│   │   │   └── ...
│   │   └── val/                   # Imagens validação
│   ├── labels/                    # Anotações YOLO para imagens
│   │   ├── train/                 # Labels treino (.txt)
│   │   └── val/                   # Labels validação (.txt)
│   ├── captura_imagens.py         # Script para captura das imagens
│   └── sacolas.yaml               # Configuração dataset YOLO
├── modelos/                      # Modelos treinados e pesos
│   └── detector_sacola.pt         # Modelo YOLO customizado
├── runs/                         # Resultados e logs de treino
│   └── train/
│       └── detector_sacola/
│           ├── weights/
│           │   ├── best.pt
│           │   └── best.onnx
│           ├── confusion_matrix.png
│           ├── results.csv
│           └── ...
├── src/                         # Código fonte da aplicação
│   ├── core/                    # Lógica da inspeção e regras de negócio
│   │   └── inspector.py
│   ├── detector/                # Detector YOLO e wrappers
│   │   └── detector.py
│   ├── ui/                      # Interface gráfica (PyQt5)
│   │   ├── main_window/         # Layouts e janelas principais
│   │   │   ├── layout.py
│   │   │   └── window.py
│   │   ├── utils/               # Utilitários da interface
│   │   │   └── styles.py
│   │   ├── views/               # Views específicas
│   │   │   ├── camera.py
│   │   │   ├── companies.py
│   │   │   ├── history.py
│   │   │   └── reports.py
│   │   └── widgets/             # Widgets customizados
│   │       ├── alerts.py
│   │       ├── company_form.py
│   │       └── sidebar.py
│   └── main.py                 # Ponto de entrada da aplicação
├── requirements.txt             # Dependências Python
├── config.yaml                  # Configurações gerais
├── Etapas.md                   # Documentação das etapas do projeto
├── README.md                   # Este arquivo
├── run.py                      # Script para executar aplicação
├── teste.py                    # Script para testes pontuais
├── train.py                    # Script para treinamento do modelo
└── verify_model.py             # Verificação e avaliação do modelo treinado                 # Script principal para iniciar o sistema
```

**Explicação:**
A arquitetura modular separa claramente as áreas funcionais, garantindo escalabilidade e facilidade de manutenção. O diretório `src` concentra toda a lógica do sistema. O módulo `core` cuida das regras de negócio e aplicação das políticas de aceitação, enquanto o `detector` encapsula o modelo de visão computacional para facilitar atualizações e treinamentos futuros. A interface gráfica está organizada para suportar múltiplas views e componentes reutilizáveis. A separação dos dados de cadastro e modelos garante flexibilidade e segurança no armazenamento local.

---

### 4. Tecnologias Utilizadas

| Categoria            | Tecnologias                                        |
| -------------------- | -------------------------------------------------- |
| Linguagem            | Python 3.10+                                       |
| Visão Computacional  | Bibliotecas consolidadas (OpenCV, PyTorch, YOLOv8) |
| Interface Gráfica    | PyQt5 com QSS para estilização                     |
| Ambiente de Execução | Windows (instalador executável fornecido)          |

---

### 5. Instalação e Execução

- Entrega via instalador Windows completo, com todas as dependências pré-configuradas.
- Execução simplificada para ambientes que atendam os requisitos mínimos de hardware.
- Suporte para uso com webcam ou câmeras industriais compatíveis.

---

### 6. Licença e Propriedade

- Licença **proprietária e exclusiva** para a empresa contratante.
- Entrega do sistema executável com todos os direitos de uso integral.
- Desenvolvedor **não mantém qualquer direito sobre o sistema entregue**.
- Reutilização da base técnica e algorítmica é permitida **apenas para outros projetos não concorrentes** e com propósitos distintos.

---

### 7. Suporte Técnico e Manutenção

- **Período de suporte técnico:** 12 meses a partir da entrega final.
- Serviços inclusos:

  - Correção de bugs críticos
  - Suporte operacional e esclarecimento de dúvidas
  - Ajustes pontuais para garantir o funcionamento contínuo

- Demandas fora do escopo serão tratadas como serviços complementares, mediante orçamento.

---

### 8. Cronograma e Prazos

| Fase                            | Prazo Estimado                              |
| ------------------------------- | ------------------------------------------- |
| Desenvolvimento e montagem      | Até 12 meses após assinatura                |
| Coleta de dados e testes locais | Conforme cronograma do cliente              |
| Entrega final com instalador    | Após validação conjunta das funcionalidades |

---

### 9. Condições Gerais

- Desenvolvimento sob medida, com acompanhamento técnico constante.
- Proibição total de distribuição ou exposição a terceiros sem autorização formal.
- Contrato formal contemplará cláusulas específicas de exclusividade, propriedade intelectual, prazos e responsabilidades.

---

### Hardware Recomendado (Configuração Indicada)

| Componente     | Modelo e Detalhes                                                |
| -------------- | ---------------------------------------------------------------- |
| Processador    | Intel Core i7-14700K, 20-Core, 28-Threads, 3.4GHz (Turbo 5.6GHz) |
| Cooler         | Water Cooler Pichau Lunara ARGB, 240mm                           |
| Placa Mãe      | Gigabyte B760M Aorus Elite DDR5 (M-ATX)                          |
| Memória RAM    | 2x 16GB Team Group T-Force Delta A RGB DDR5 5600MHz              |
| Armazenamento  | SSD Kingston NV3 1TB PCIe NVMe (Leitura 6000MB/s)                |
| Placa de Vídeo | MSI GeForce RTX 5060 Ti Gaming OC 16GB GDDR7                     |
| Fonte          | Pichau Cluster 750W Full Modular Cybenetics Gold                 |
| Gabinete       | Mancer CV500L Mid Tower com vidro lateral                        |
| Acessórios     | Cabos HDMI e de força originais Pichau                           |

---

### Contato

Para negociação, dúvidas técnicas, assinatura contratual e suporte, entre em contato diretamente:

**Lenon Yuri**
Desenvolvedor de Sistemas
© 2025
