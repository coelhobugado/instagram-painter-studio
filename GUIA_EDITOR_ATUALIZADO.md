# Editor de áreas e tintas — versão local

## Uso

1. A janela abre dentro da área visível da tela. Use **F11** para alternar a tela cheia e **Esc** para sair dela.
2. Abra uma imagem e escolha um dos quatro modos: **Automático** (decide entre linhas e cores), **Traçado fino** (centro das linhas), **Pintura em cores** (áreas e sombras) ou **Contornos** (bordas de alto contraste). Gere a prévia; a aba **Pintura** mostra o modo que foi realmente usado.
   Em **Automático** ou **Traçado fino**, marque **Não pintar o fundo (usar a tela do celular)** antes de gerar a prévia se quiser executar somente os traços. O plano criado não inclui pincel grosso de preenchimento.
3. Na aba **Pintura**, clique para selecionar uma parte ou arraste um retângulo para selecionar várias de uma vez. Use **Ctrl+clique** ou **Ctrl+arrastar** para adicionar partes separadas. A borda amarela marca a seleção. Para detalhes pequenos, use **Ampliar**, que também aceita seleção por arrasto. Use **Selecionar fundo** para editar o fundo inteiro explicitamente; cliques em áreas vazias não o selecionam. **Limpar seleção** desmarca as partes atuais.
4. Use **Alterar cor da seleção** para escolher um RGB e obter a tinta física mais próxima, ou **Tintas do Instagram** para selecionar diretamente uma amostra do perfil nas páginas 1, 2 ou 3. Todas as partes selecionadas recebem a cor em uma única edição, que pode ser desfeita de uma vez.
5. Use **Ver traços** para inspecionar os eixos individuais. **Ver pintura** mostra a espessura estimada pelo perfil do aparelho.
6. **Desfazer** e **Refazer** mantêm até cinco revisões na sessão. **Salvar projeto** guarda um arquivo `.npz` editável; **Abrir projeto** reconstrói a prévia e todas as verificações.

Ao trocar imagem, modo, cor, quantidade de tintas ou área do celular, a prévia anterior fica desatualizada e o início da pintura é bloqueado até gerar outra prévia.

O botão de início e o progresso ficam no rodapé, sempre visíveis. O início só é liberado com prévia aprovada e celular conectado. O painel de logs começa recolhido e pode ser expandido quando necessário. A aba **Qualidade** resume se a prévia foi aprovada e explica as medidas e o mapa de erro; a aba **Cores** mostra uma linha compacta por tinta.

Os cartões do inspetor de paleta continuam alterando todas as áreas daquela entrada de cor. A edição pela prévia muda somente as partes selecionadas. Regiões sem traços planejados não são selecionáveis.

## Garantias locais

- A identidade da tinta inclui página, gesto e coordenadas. Uma amostra da página 2 não se confunde com uma da página 1 no mesmo ponto da tela.
- As edições criam outra revisão e recompilam gestos, prévia, alvo executável, diagnóstico, tempo e verificação de qualidade. Um erro mantém a revisão anterior.
- O executor exige que a mudança de posição do controle de espessura seja visível antes de pintar. Se a captura não confirmar a transição, ele bloqueia o primeiro traço daquela camada.
- Depois de cada camada, a conferência tenta alinhar a captura à prévia. Quando o fundo da conversa impede esse alinhamento, os traços finos são conferidos pela cor e pela mudança nos caminhos e coordenadas planejados; um fundo uniforme é conferido pela área e pela cor pintadas. Evidência insuficiente ainda interrompe a execução e preserva o checkpoint.
- A prévia usa a mesma simplificação dos caminhos enviada pela via ADB tradicional. O modo **Ver traços** permite enxergar linhas centrais que se sobrepõem na pintura estimada.

## Limite físico ainda pendente

A verificação visual confirma que o controle lateral mudou de posição; ela não mede a largura real do traço no Instagram. O perfil atual prevê 10 px no mínimo, 52 px no médio e 98 px no grosso. Essa calibração, a posição exata do slider e a fidelidade dos gestos longos em páginas laterais precisam de teste no celular quando ele estiver disponível. A execução física não foi realizada nesta atualização.

## Limite da seleção

Em **Traçado fino** e **Contornos**, o retângulo seleciona todos os traços que cruzam a área; cada traço selecionado muda por inteiro. Em **Pintura em cores**, ele seleciona os componentes de cor que tocam a área. Partes conectadas da mesma cor podem aparecer como uma área só. Corrigir máscaras manualmente e medir o resultado físico ainda não são suportados.
