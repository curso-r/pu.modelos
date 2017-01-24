---
title: Modelagem
date: '2017-01-24'
---





## Aprendizado Estatístico

O termos *Aprendizado Estatístico* refere-se a uma vasta quantidade de ferramentas
que são utilizadas para entender dados. Essas ferramentas são classificadas em 
**supervisionadas** e **não-supervisionadas**. De forma geral, aprendizado
supervisionado envolve a construção de um modelo estatístico para prever ou estimar
uma **resposta** de acordo com uma ou mais informações de entrada. No aprendizado 
não-supervisionado existem variáveis de entrada mas não existe uma variável resposta. 
Neste caso, o objetivo é entender a estrutura e a relação entre as variáveis. Existe
uma terceira classificação para as ferramentas de aprendizado estatístico chamada 
[Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning), mas
não abordaremos este tema neste material.

### Exemplos

1. Um estudo estatístico cujo objetivo é estimar a probabilidade de uma transação 
ser uma fraude e são fornecidos dados relativos a transações passadas bem como se 
estas foram uma fraude ou não. É considerado um estudo de aprendizado supervisionado.

2. Um estudo em que são fornecidas diversas informações sobre os hábitos de compras
dos clientes e deseja-se identificar diferentes segmentos, é um
estudo de aprendizado não-supervisionado.

---------

Neste material vamos abordar incialmente algumas técnicas de aprendizado supervisionado.
Em seguida abordaremos abordaremos superficialmente alguns conceitos de aprendizado 
não-supervisionado. Todos esses conceitos serão apresentados com exemplos práticos 
usando o R. 

Para se aprofundar mais no assunto os seguintes links são ótimas referências.

* [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Sixth%20Printing.pdf)
* [Coursera - Practical Machine Learning](https://www.coursera.org/learn/practical-machine-learning)








## Aprendizado Supervisionado

Suponha que você observou uma variável resposta $Y$ e $p$ diferentes variáveis 
explicativas $X_1, X_2, ..., X_p$. Assumimos que existe alguma relação entre $Y$
e $X = (X_1, X_2, ..., X_p)$. Podemos denotar matematicamente esta relação como
na seguinte equação:

$$Y = f(X) + \epsilon$$

O objetivo geral do aprendizado supervisionado é estimar a função $f$.
Nessa formulação, $\epsilon$ é um termo de erro aleatório com média 0. $f$ representa
a informação sistemática que $X$ fornece sobre $Y$.

### Modelos lineares

O modelo linear assume que a função $f$ é uma função linear de modo que a formulação
do apredizado supervisionado pode ser reescrita da seguinte forma:

$$Y = \alpha + X\beta + \epsilon$$
Em que $\alpha$ e $\beta$ são coeficientes que serão estimados. Esses valores são 
calculados de forma a minimizar uma **função de perda** na sua amostra. A função 
mais utilizada é a perda quadrática na sua amostra. Considere $(y_1, x_1)$, $(y_2, x_2)$, ..., $(y_n, x_n)$ uma amostra de tamanho $n$.

$\alpha$ e $\beta$ são escolhidos de tal forma que:

$$\sum_{i = 1}^{n} [y_i - (\alpha + \beta x_i)]^2$$
seja o menor possível. Isto é, estamos minimizando o *erro quadrático*.

Na ótica da estatística, assumimos também que $Y \sim Normal(\alpha + X \beta, \sigma^2)$, 
escolhemos $\alpha$ e $\beta$ de forma que maximize uma função que chamamos de [verossimilhança](https://pt.wikipedia.org/wiki/Fun%C3%A7%C3%A3o_de_verossimilhan%C3%A7a). 
Essa suposição é útil quando queremos fazer testes de hipóteses e intervalos de 
confiança. Por enquanto, não estamos interessados nisso e portanto vamos 
apresentar uma visão menos complexa.

#### Exemplo

Considere o banco de dados *BodyFat* obtido [aqui](http://www2.stetson.edu/~jrasp/data.htm). 
Esses são dados do percentual de gordura corporal em uma amostra de 252 homens junto com
diversas outras medidas corporais. O percentual de gordura corporal é medido pesando
a pessoa sob a água, um procedimento trabalhoso. O objetivo é fazer um modelo linear
que permita obter o percentual de gordura usando medidas do corpo fáceis de serem obtidas.
Os dados são do site do Journal of Statistics Education.


```r
library(readxl)
library(dplyr)
library(ggplot2)
bodyfat <- read_excel('data/BodyFat.xls')
```


```r
ggplot(bodyfat, aes(x = WEIGHT, y = BODYFAT)) + geom_point()
```

<img src="figures//unnamed-chunk-7-1.png" title="plot of chunk unnamed-chunk-7" alt="plot of chunk unnamed-chunk-7" width="50%" height="50%" />

A partir do gráfico de dispersão, vemos que o peso do indivíduo parece ser **linearmente**
relacionado ao percentual de gordura corporal. Vamos então ajustar um modelo linear
usando o R. Para ajustar o modelo, usamos a função `lm` (de *__l__inear __m__odel*). 
A função `lm`, assim como muitas outras que ajustam modelo no R, recebe como argumentos
uma formula e um banco de dados. 

`formula` é um tipo especial de objeto no R que ajuda muito na especificação dos modelos. 
Ela tem a forma `y ~ x1 + x2 + ... + xn` em que `y` é o nome da variável resposta e `x1`,
`x2`, ..., `xn` são os nomes das variáveis que serão utilizadas como explicativas. 


```r
ajuste <- lm(BODYFAT ~ WEIGHT, data = bodyfat)
```

Com essa chamada da função criamos o objeto `ajuste`. Esse objeto abriga informações
relacionadas ao ajuste do modelo.

$$bodyfat = \alpha + \beta*weight + \epsilon$$
As estimativas de $\alpha$ e $\beta$ podem ser encontradas usando a função `summary`.
A estimativa de $\alpha$ é o valor da coluna `Estimate` na linha `(Intercept)`: -9.99515 
e a estimativa de $\beta$ é o valor logo abaixo: 0.16171.


```r
summary(ajuste)
## 
## Call:
## lm(formula = BODYFAT ~ WEIGHT, data = bodyfat)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -16.434  -4.315   0.079   4.540  19.681 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -9.99515    2.38906  -4.184 3.97e-05 ***
## WEIGHT       0.16171    0.01318  12.273  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 6.135 on 250 degrees of freedom
## Multiple R-squared:  0.376,	Adjusted R-squared:  0.3735 
## F-statistic: 150.6 on 1 and 250 DF,  p-value: < 2.2e-16
```

Em R, o ajuste de um modelo estatístico é salvo em um objeto. Esse objeto é uma
`list` que armazena diversas informações sobre o ajuste. Você pode ver algumas
informações disponíveis quando vê a estrutura do objeto `ajuste` usando a função 
`str`.


```r
str(ajuste, max.level = 1)
## List of 12
##  $ coefficients : Named num [1:2] -9.995 0.162
##   ..- attr(*, "names")= chr [1:2] "(Intercept)" "WEIGHT"
##  $ residuals    : Named num [1:252] -2.35 -11.12 9.69 -8.98 8 ...
##   ..- attr(*, "names")= chr [1:252] "1" "2" "3" "4" ...
##  $ effects      : Named num [1:252] -300.64 75.29 10.38 -9.01 7.98 ...
##   ..- attr(*, "names")= chr [1:252] "(Intercept)" "WEIGHT" "" "" ...
##  $ rank         : int 2
##  $ fitted.values: Named num [1:252] 14.9 18 14.9 19.9 19.8 ...
##   ..- attr(*, "names")= chr [1:252] "1" "2" "3" "4" ...
##  $ assign       : int [1:2] 0 1
##  $ qr           :List of 5
##   ..- attr(*, "class")= chr "qr"
##  $ df.residual  : int 250
##  $ xlevels      : Named list()
##  $ call         : language lm(formula = BODYFAT ~ WEIGHT, data = bodyfat)
##  $ terms        :Classes 'terms', 'formula'  language BODYFAT ~ WEIGHT
##   .. ..- attr(*, "variables")= language list(BODYFAT, WEIGHT)
##   .. ..- attr(*, "factors")= int [1:2, 1] 0 1
##   .. .. ..- attr(*, "dimnames")=List of 2
##   .. ..- attr(*, "term.labels")= chr "WEIGHT"
##   .. ..- attr(*, "order")= int 1
##   .. ..- attr(*, "intercept")= int 1
##   .. ..- attr(*, "response")= int 1
##   .. ..- attr(*, ".Environment")=<environment: 0x2d11688> 
##   .. ..- attr(*, "predvars")= language list(BODYFAT, WEIGHT)
##   .. ..- attr(*, "dataClasses")= Named chr [1:2] "numeric" "numeric"
##   .. .. ..- attr(*, "names")= chr [1:2] "BODYFAT" "WEIGHT"
##  $ model        :'data.frame':	252 obs. of  2 variables:
##   ..- attr(*, "terms")=Classes 'terms', 'formula'  language BODYFAT ~ WEIGHT
##   .. .. ..- attr(*, "variables")= language list(BODYFAT, WEIGHT)
##   .. .. ..- attr(*, "factors")= int [1:2, 1] 0 1
##   .. .. .. ..- attr(*, "dimnames")=List of 2
##   .. .. ..- attr(*, "term.labels")= chr "WEIGHT"
##   .. .. ..- attr(*, "order")= int 1
##   .. .. ..- attr(*, "intercept")= int 1
##   .. .. ..- attr(*, "response")= int 1
##   .. .. ..- attr(*, ".Environment")=<environment: 0x2d11688> 
##   .. .. ..- attr(*, "predvars")= language list(BODYFAT, WEIGHT)
##   .. .. ..- attr(*, "dataClasses")= Named chr [1:2] "numeric" "numeric"
##   .. .. .. ..- attr(*, "names")= chr [1:2] "BODYFAT" "WEIGHT"
##  - attr(*, "class")= chr "lm"
```

Por exemplo você pode acessar os coeficientes do modelo usando `ajuste$coefficients`.

Outra função que existe para a maior parte dos modelos que podem ser ajustados usando o R 
a `predict`. Usamos a função `predict` para obter as estimativas do modelo ajustado para
uma base de dados (nova ou não).


```r
bodyfat$predito_modelo1 <- predict(ajuste, newdata = bodyfat)
bodyfat %>% select(WEIGHT, BODYFAT, predito_modelo1) %>% head() %>% knitr::kable()
```



| WEIGHT| BODYFAT| predito_modelo1|
|------:|-------:|---------------:|
| 154.25|    12.6|        14.94842|
| 173.25|     6.9|        18.02089|
| 154.00|    24.6|        14.90800|
| 184.75|    10.9|        19.88054|
| 184.25|    27.8|        19.79969|
| 210.25|    20.6|        24.00412|

Nessa tabela, vemos o valor predito pelo modelo para cada observação bem como o
valor verdadeiro de gordura corporal daquele indivíduo. Nosso modelo não parece
estar muito bom. Uma possível medida de erro é o MSE (Erro quadrático médio).
Podemos calculá-lo fazendo contas simples no R.


```r
mse <- mean((bodyfat$BODYFAT - bodyfat$predito_modelo1)^2)
mse
## [1] 37.34089
```

É mais fácil identificar se esse erro é baixo ou não comparando-o com o erro se 
usássemos a média da variável como valor predito para todas as observações e 
tirando a raíz quadrada dos dois.


```r
erro_usando_media <- mean((bodyfat$BODYFAT - mean(bodyfat$BODYFAT))^2)
erro_usando_media
## [1] 59.83737

sqrt(mse)
## [1] 6.110719
sqrt(erro_usando_media)
## [1] 7.735462
```

Agora podemos ter uma ideia de que o nosso erro está alto. Se usássemos apenas a 
média erraríamos em média 7,7 usando o nosso modelo, ficamos com 6,1.

Felizmente, podemos melhorar o modelo adicionando mais variáveis. No R basta:


```r
ajuste2 <- lm(BODYFAT ~ WEIGHT + HEIGHT + CHEST + ABDOMEN + NECK + KNEE, 
              data = bodyfat)
```

O erro pode ser novamente calculado repetindo as operações que fizemos anteriormente.


```r
bodyfat$predito_modelo2 <- predict(ajuste2, newdata = bodyfat)
mse <- mean((bodyfat$BODYFAT - bodyfat$predito_modelo2)^2)
sqrt(mse)
## [1] 4.049453
```

Agora reduzimos bastante o erro. É muito importante ressaltar que estamos avaliando
o erro dentro da mesma base de dados que utilizamos para ajustar o modelo. Isso é 
considerado uma má prática, pois podemos facilmente esbarrar em uma situação de
*superajuste* ou *overfitting*.

----------------

Até agora vimos que usando a função `lm` podemos ajustar um modelo linear usando o
R. Esse único comando, que recebe um formula e um banco de dados, retorna um objeto 
que é similar a uma `list` e que armazena uma variedade de informações sobre o 
ajuste como coeficientes, dados utilizados, etc. Aprendemos também a função `summary`, 
que "imprime" no console uma série de informações sobre o ajuste. Também vimos a 
função `predict` que é utilizada pra obter os valores preditos pelo modelo para 
uma nova base de dados.

Mais tarde falaremos novamente sobre modelos lineares quando falarmos sobre 
[regressão logística](https://pt.wikipedia.org/wiki/Regress%C3%A3o_log%C3%ADstica).

### Árvore  de Decisão











<script src="https://cdn.datacamp.com/datacamp-light-latest.min.js"></script>




<script src="https://cdn.datacamp.com/datacamp-light-latest.min.js"></script>



1. Calcule o número de ouro no R.

$$
\frac{1 + \sqrt{5}}{2}
$$

<div data-datacamp-exercise data-height="300" data-encoded="true">eyJsYW5ndWFnZSI6InIiLCJzYW1wbGUiOiIjIERpZ2l0ZSBhIGV4cHJlc3NcdTAwZTNvIHF1ZSBjYWxjdWxhIG8gblx1MDBmYW1lcm8gZGUgb3Vyby4iLCJzb2x1dGlvbiI6IigxICsgc3FydCg1KSkvMiIsInNjdCI6InRlc3Rfb3V0cHV0X2NvbnRhaW5zKFwiMS42MTgwMzRcIiwgaW5jb3JyZWN0X21zZyA9IFwiVGVtIGNlcnRlemEgZGUgcXVlIGluZGljb3UgYSBleHByZXNzXHUwMGUzbyBjb3JyZXRhbWVudGU/XCIpXG5zdWNjZXNzX21zZyhcIkNvcnJldG8hXCIpIn0=</div>






