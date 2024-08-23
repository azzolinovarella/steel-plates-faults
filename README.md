# Steel Plates Faults
O presente repositório contém todo o código elaborado para o projeto final da disciplina Aprendizado de Máquina I da terceira edição do Programa de Especialização em Software (PES 2024) da Embraer, lecionada pelo professor George Darmiton da Cunha Cavalcanti.

Este projeto utilizou a base de dados [Steel Plates Faults](https://www.openml.org/search?type=data&status=active&id=1504&sort=runs) do repositório de aprendizado de máquina da UCI. O conjunto de dados contém 1.941 instâncias de falhas em placas de aço, classificadas em 7 tipos diferentes. Ele possui 27 características independentes que são usadas para tarefas de classificação. As variáveis incluem medidas como mínimos e máximos em X e Y, áreas de pixels, perímetros e luminosidade. Assim, o objetivo deste trabalho é dar continuidade ao estudo iniciado na disciplina anterior, Aprendizagem de Máquina I, aplicando e aprofundando os conhecimentos adquiridos por meio da introdução de novas técnicas, como o balanceamento de dados e a combinação de classificadores.

## Relatório Experimental
Informações mais detalhadas sobre o desenvolvimento do projeto podem ser encontradas no relatório presente em [`./docs/relatorio-final-ami.pdf`](./docs/relatorio-final-amii.pdf). 

## Explicação da estrutura do projeto
O presente projeto possuí a seguinte estrutura:

&nbsp; . <br>
├── `data` -> Contém todos os dados utilizados neste projeto, incluindo a base de dados original, a base de dados tratada, o mapa dos targets para o nome das classes e o mapa do id das colunas para o seu respectivo nome. <br>
├── `docs` -> Contém o relatório desenvolvido neste projeto, as instruções passadas pelo professor e o esquemático do procedimento adotado. <br>
├── `figs` -> Contém todas as figuras geradas nos notebooks. <br>
├── `notebooks` -> Contém os notebooks de tratamento dos dados, EDA (simplificada devido a não ser o foco deste projeto), validação dos algoritmos de treinamento (com e sem bagging) e realização de testes de hipótese para verificar melhoria nos resultados após ensemble. <br>
├── `environment.yml` -> Contém as dependências necessárias para reconstruir o ambiente virtual conda utilizado neste projeto. <br>
└── `README.md` -> Contém a descrição do projeto, explicação da estrutura e dos dados utilizados. <br>

## Autores
- Felipe Azzolino Varella - [@azzolinovarella](https://github.com/azzolinovarella) (fav3@cin.ufpe.br | f.azzolinovarella@gmail.com)
- Matheus Augusto Ladeira Cayres - [@latheusmadeira](https://github.com/latheusmadeira) (malc@cin.ufpe.br | maurin-96@hotmail.com)
