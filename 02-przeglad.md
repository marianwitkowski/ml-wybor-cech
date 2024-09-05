## 2. **Przegląd Metod Wyboru Cech**

Wybór odpowiednich cech jest kluczowym etapem w procesie budowy modeli uczenia maszynowego. Może znacząco wpłynąć na wydajność, dokładność i interpretowalność modelu. W tej części omówimy szczegółowo różne metody wyboru cech, w tym te oparte na filtrach, osadzaniu oraz przeszukiwaniu. Każda z tych metod zostanie zilustrowana odpowiednimi wzorami i przykładami kodu w Pythonie.

#### Selekcja cech oparta na filtrach

Selekcja cech oparta na filtrach polega na ocenie cech niezależnie od modelu uczenia maszynowego. Cechy są oceniane na podstawie statystyk lub innych miar, a następnie wybierane są te, które najlepiej spełniają określone kryteria.

##### Statystyczne testy istotności

Statystyczne testy istotności pomagają ocenić, czy istnieje istotny związek między cechą a zmienną docelową. Oto kilka popularnych testów statystycznych stosowanych w tym celu:

1. **Test t-Studenta**

Test t-Studenta porównuje średnie dwóch grup, aby sprawdzić, czy różnią się one w sposób statystycznie istotny. Jest to przydatne, gdy mamy zmienną binarną jako zmienną docelową.

Wzór na statystykę t:

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
$$

Gdzie:
- $\bar{X}_1$ i $\bar{X}_2$ to średnie wartości dla dwóch grup,
- $s_1^2$ i $s_2^2$ to wariancje w tych grupach,
- $n_1$ i $n_2$ to liczba obserwacji w każdej grupie.

Przykładowy kod w Pythonie:

```python
import pandas as pd
from scipy import stats

# Załóżmy, że mamy DataFrame z cechą "feature" i zmienną docelową "target"
df = pd.DataFrame({
    'feature': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

# Podział na dwie grupy na podstawie zmiennej docelowej
group1 = df[df['target'] == 0]['feature']
group2 = df[df['target'] == 1]['feature']

# Test t-Studenta
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"T-statystyka: {t_stat}, Wartość p: {p_value}")
```

2. **ANOVA (Analysis of Variance)**

ANOVA porównuje średnie wartości cechy między więcej niż dwoma grupami. Wzór na statystykę F (stosowaną w ANOVA):

$$
F = \frac{\text{SSB}/k-1}{\text{SSW}/n-k}
$$

Gdzie:
- SSB to suma kwadratów między grupami,
- SSW to suma kwadratów wewnątrz grup,
- $k$ to liczba grup,
- $n$ to całkowita liczba obserwacji.

Przykładowy kod w Pythonie:

```python
from sklearn.feature_selection import f_classif

# Załóżmy, że mamy cechy w X i zmienną docelową w y
X = df[['feature']]
y = df['target']

# ANOVA
F_stat, p_values = f_classif(X, y)
print(f"F-statystyka: {F_stat}, Wartości p: {p_values}")
```

3. **Chi-kwadrat**

Test chi-kwadrat jest używany do badania zależności między dwiema zmiennymi kategorialnymi. Wzór na statystykę chi-kwadrat:

$$
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
$$

Gdzie:
- $O_i$ to obserwowana częstość,
- $E_i$ to oczekiwana częstość.

Przykładowy kod w Pythonie:

```python
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# Normalizacja cech, aby zapewnić dodatnie wartości dla testu chi-kwadrat
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Test chi-kwadrat
chi2_stat, p_values = chi2(X_norm, y)
print(f"Chi-kwadrat statystyka: {chi2_stat}, Wartości p: {p_values}")
```

##### Metody miar podobieństwa (np. korelacja)

Korelacja jest często stosowana w selekcji cech do oceny liniowego związku między cechami a zmienną docelową.

1. **Korelacja Pearsona**

Korelacja Pearsona mierzy liniową zależność między dwiema zmiennymi. Wzór na współczynnik korelacji Pearsona:

$$
r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}
$$

Przykładowy kod w Pythonie:

```python
# Korelacja Pearsona
correlation_matrix = df.corr()
print(correlation_matrix)
```

2. **Korelacja Spearmana**

Korelacja Spearmana jest miarą monotonicznej zależności między zmiennymi. Wzór na współczynnik korelacji Spearmana:

$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

Gdzie $d_i$ to różnica w rangach dla każdej pary obserwacji.

Przykładowy kod w Pythonie:

```python
# Korelacja Spearmana
spearman_corr = df.corr(method='spearman')
print(spearman_corr)
```

3. **Korelacja Kendalla**

Korelacja Kendalla mierzy zgodność porządkową między dwiema zmiennymi.

Przykładowy kod w Pythonie:

```python
# Korelacja Kendalla
kendall_corr = df.corr(method='kendall')
print(kendall_corr)
```

#### Selekcja cech oparta na osadzaniu (embedded)

Metody osadzania integrują selekcję cech bezpośrednio z procesem trenowania modelu, co pozwala na wybór cech, które są najbardziej istotne dla predykcji. Popularne techniki obejmują drzewa decyzyjne, lasy losowe oraz regularizację (LASSO i Ridge).

##### Drzewa decyzyjne i lasy losowe

Drzewa decyzyjne oceniają cechy na podstawie ich wpływu na redukcję nieczystości (np. entropii lub indeksu Gini) w każdym węźle drzewa. Lasy losowe, będące zespołem drzew decyzyjnych, obliczają średnią wagę dla każdej cechy na podstawie jej wkładu w różne drzewa.

Przykładowy kod w Pythonie:

```python
from sklearn.ensemble import RandomForestClassifier

# Inicjalizacja modelu lasów losowych
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Trenowanie modelu
model.fit(X, y)

# Ważność cech
feature_importances = model.feature_importances_
print(feature_importances)
```

##### Regularyzacja (LASSO, Ridge)

Regularyzacja penalizuje wagi cech w modelach regresyjnych, co zapobiega nadmiernemu dopasowaniu. LASSO (L1) ma zdolność do zerowania wag nieistotnych cech, natomiast Ridge (L2) zmniejsza wszystkie wagi, ale nie zeruje ich.

1. **LASSO (L1 regularization)**

LASSO dodaje karę opartą na sumie wartości bezwzględnych wag:

$$
\text{Funkcja straty} = \text{RSS} + \lambda \sum |w_i|
$$

Gdzie:
- RSS to suma kwadratów reszt,
- $\lambda$ to parametr regularizacji,
- $w_i$ to wagi cech.

Przykładowy kod w Pythonie:

```python
from sklearn.linear_model import Lasso

# Inicjalizacja modelu LASSO z regularyzacją L1
lasso = Lasso(alpha=0.1)

# Trenowanie modelu
lasso.fit(X, y)

# Ważność cech (

współczynniki)
print(lasso.coef_)
```

2. **Ridge (L2 regularization)**

Ridge dodaje karę opartą na sumie kwadratów wag:

$$
\text{Funkcja straty} = \text{RSS} + \lambda \sum w_i^2
$$

Przykładowy kod w Pythonie:

```python
from sklearn.linear_model import Ridge

# Inicjalizacja modelu Ridge z regularyzacją L2
ridge = Ridge(alpha=1.0)

# Trenowanie modelu
ridge.fit(X, y)

# Ważność cech (współczynniki)
print(ridge.coef_)
```

##### Elastic Net

Elastic Net łączy w sobie zalety LASSO i Ridge, stosując oba rodzaje regularyzacji.

Przykładowy kod w Pythonie:

```python
from sklearn.linear_model import ElasticNet

# Inicjalizacja modelu Elastic Net
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)

# Trenowanie modelu
elastic_net.fit(X, y)

# Ważność cech (współczynniki)
print(elastic_net.coef_)
```

#### Selekcja cech oparta na przeszukiwaniu (wrapper)

Metody oparte na przeszukiwaniu polegają na iteracyjnym przeszukiwaniu przestrzeni cech w celu znalezienia optymalnego podzbioru cech, który maksymalizuje wydajność modelu.

##### Metody do przeszukiwania przestrzeni cech (np. RFE)

Rekurencyjna Eliminacja Cech (RFE) to jedna z popularnych metod przeszukiwania. RFE iteracyjnie trenuje model i usuwa najmniej istotne cechy, aż do osiągnięcia pożądanego zestawu cech.

Przykładowy kod w Pythonie:

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Model bazowy
model = LogisticRegression()

# RFE
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X, y)

# Wyniki RFE
print("Selected Features:", fit.support_)
print("Feature Ranking:", fit.ranking_)
```

##### Cross-validation w wyborze cech

Cross-validation jest techniką walidacyjną, która polega na podziale danych na kilka foldów, trenowaniu modelu na jednym z nich i testowaniu na innych. Można ją zastosować w selekcji cech, aby zapewnić, że wybrane cechy rzeczywiście poprawiają wydajność modelu.

Przykładowy kod w Pythonie:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Model bazowy
model = LogisticRegression()

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Średni wynik walidacji krzyżowej: {scores.mean()}")
```

Cross-validation może być również stosowana w połączeniu z RFE:

```python
from sklearn.feature_selection import RFECV

# RFECV - RFE z walidacją krzyżową
rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
rfecv.fit(X, y)

print(f"Optymalna liczba cech: {rfecv.n_features_}")
print(f"Wagi cech: {rfecv.support_}")
```

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone
