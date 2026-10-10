# ComfyUI TriFlow

<img width="1391" height="663" alt="image" src="https://github.com/user-attachments/assets/320c7780-8917-4c16-9be4-30af5b862774" />

Um único custom node **TriFlow Remesh**, com entrada `MESH` e saída `MESH` no
formato nativo do ComfyUI. Gera uma nova topologia usando o modelo TriFlow,
os VAEs SDF/NVV e reconstrução por QEM. Os scripts, configurações e fontes
necessários estão neste projeto;

## Instalação

Coloque esta pasta em `ComfyUI/custom_nodes/comfy_ui_triflow` ou clone:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SaluRamos/ComfyUI_Triflow.git comfy_ui_triflow
cd comfy_ui_triflow
python install.py
```

Use **o mesmo Python que executa o ComfyUI** para chamar `install.py`.

### Windows: ambiente próprio do node

O instalador cria `.python/` (Python 3.11) e `.venv/` dentro desta pasta.
PyTorch 2.6/CUDA 12.4, MeshLib, Meshiki, spconv e o QEM ficam nesse ambiente,
sem trocar o PyTorch ou as bibliotecas do Python principal do ComfyUI.
O QEM compilado para Python 3.11/Windows x64 está em `wheels/`. Se esse arquivo
não estiver presente, o instalador usa as ferramentas C++ do Visual Studio e,
quando necessário, baixa um SDK da Microsoft para `.installer/sdk/`.

O node executa um processo Python Windows separado e troca malhas OBJ temporárias.
Não usa WSL nem a instalação em `repos/triflow`. Cancelar o job encerra o processo.
O backend Windows usa o algoritmo nativo do spconv com conversão dos kernels originais TorchSparse,
mantendo a ordem dos eixos e os voxels ativos. A atenção usa o SDPA do PyTorch;
não exige instalar TorchSparse, FlashAttention ou Triton no Windows.

```powershell
& "CAMINHO/ComfyUI/.venv/Scripts/python.exe" "CAMINHO/comfy_ui_triflow/install.py"
```

Reinicie o ComfyUI após a instalação. Entrada, saída e controles do node permanecem iguais.

### Linux

O instalador instala as dependências no Python ativo. O runtime Linux original
usa TorchSparse, spconv, FlashAttention e Triton; instale os backends CUDA
compatíveis com seu PyTorch. O ambiente original usa Python 3.10,
PyTorch 2.4 e CUDA 11.8.

### Pesos

Ao executar o node, os arquivos ausentes são baixados automaticamente de
[lihcxr/TriFlow](https://huggingface.co/lihcxr/TriFlow). Arquivos já presentes
não são baixados novamente. No Comfy Desktop para Windows, a pasta usada é:

```text
C:\Users\SaluC\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\triflow
```

Em outras instalações, usa `ComfyUI/models/triflow/`. Os três pesos são
`flow_model.safetensors`, `sdf_vae.safetensors` e `nvv_vae.safetensors`
(aproximadamente 1,32 GB). Downloads incompletos não são usados como modelos.
Reinicie o ComfyUI depois de atualizar o node.

## Uso

Procure **TriFlow Remesh**, categoria `3d/TriFlow`, e conecte:

```text
node que produz MESH → TriFlow Remesh → Save 3D Model / SaveGLB
```

| Controle | Padrão | Efeito |
| --- | --- | --- |
| `qem_threshold` | `12.0` | Limite de erro QEM; influencia o quanto a geometria pode ser simplificada. |
| `face_count` | `4000` | Contagem alvo de triângulos. `0` usa a contagem da entrada após o pré-processamento. |
| `quad_ratio` | `0.95` | Condição de topologia do modelo, de `0` a `1`. `0` é uma razão válida. |

O nome correto é `qem_threshold`. A contagem de faces é aproximada: os limites
de erro e as restrições de topologia podem impedir atingir o alvo. `quad_ratio`
influencia a topologia aprendida; **a saída continua sendo triangular**.

Aceita meshes nativos com tensores `(B, N, 3)` e `(B, F, 3)`, incluindo lotes
com padding e `vertex_counts`/`face_counts`. Posição e escala são restauradas
após a inferência. Lotes com saídas de tamanhos diferentes exigem ComfyUI com
`pack_variable_mesh_batch`.

`TRIMESH` de extensões externas precisa ser convertido para `MESH` antes do
node. A nova topologia não preserva UVs, cores ou texturas; faça unwrap/bake
após a remalhagem. A inferência é estocástica e usa 50 passos a resolução 512,
como o pipeline original. Não expõe seed nesta primeira versão.

Os modelos anteriores do ComfyUI são descarregados da GPU antes da execução.
Os três modelos TriFlow são carregados uma vez por execução, compartilhados
entre itens do lote e liberados ao final. Não ficam retidos na GPU entre jobs.
Arquivos OBJ intermediários ficam em uma pasta temporária, removida ao final.

## Validação

Windows, com o Python privado do node:

```powershell
& ".venv/Scripts/python.exe" "scripts/check_windows.py"
```

Teste CUDA Windows concluído: esfera deslocada e escalada, alvo de 320 faces,
saída com 318 faces, coordenadas preservadas. ComfyUI manteve seu PyTorch original.


Execute com o Python do ambiente CUDA, de qualquer diretório:

```bash
python /caminho/comfy_ui_triflow/scripts/check_runtime.py
python /caminho/comfy_ui_triflow/scripts/check_runtime.py --remesh
python /caminho/comfy_ui_triflow/scripts/check_standalone.py --remesh
python -m unittest discover -s tests -v
```

`--remesh` executa inferência real numa esfera deslocada e escalada, verifica
geometria não vazia e a restauração das coordenadas. Os testes do adaptador usam
um substituto da classe `MESH` com o mesmo contrato de tensores; não precisam
de uma instalação ComfyUI nem carregam modelos CUDA.

`check_standalone.py` copia o projeto para uma pasta temporária fora da origem,
compila o QEM copiado e roda os checks usando essa cópia. Exige espaço para uma
cópia dos pesos, compilador C++ e acesso às dependências de build declaradas
no `pyproject.toml` do QEM.

## Código incluído e licenças

O código TriFlow mantém a licença **Automotive Development Public
Non-Commercial License v1.0**, em [LICENSE](LICENSE).
As cópias reduzidas de [TRELLIS](https://github.com/microsoft/TRELLIS) e
[Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2) mantêm suas licenças MIT
em `vendor/`. O fork QEM mantém sua licença em `third_party/pyfqmr-triflow/LICENSE`.
Os kernels de atenção Direct3D-S2 também conservam os avisos Apache-2.0 originais.
MeshLib é uma dependência externa e possui termos de licença próprios.

Foram removidos componentes de treinamento, renderização e geração de imagem
sem uso no pipeline de inferência. Imports e targets Hydra receberam nomes
próprios para permitir coexistência com outras extensões 3D.


## Intel Arc / Intel Arc Pro B70 (XPU)

See [INTEL_XPU.md](INTEL_XPU.md) for installation, backend selection and validation limits.
