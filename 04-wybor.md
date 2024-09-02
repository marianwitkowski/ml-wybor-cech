## 4. **Proces Wyboru Cech**

Wybór cech to jeden z najważniejszych etapów w budowie modelu uczenia maszynowego. Cechy, które dostarczamy modelowi, decydują o jego zdolności do przewidywania i generalizacji na nowe dane. Dlatego proces selekcji cech musi być przeprowadzony z najwyższą starannością. W tym rozdziale omówimy cały proces wyboru cech, począwszy od preprocessing danych, poprzez eksploracyjną analizę danych, aż po implementację różnych metod selekcji cech.

#### Preprocessing danych

Preprocessing danych to pierwszy krok w procesie wyboru cech. Na tym etapie dokonuje się wstępnego przetwarzania surowych danych, co obejmuje czyszczenie danych, normalizację, standaryzację oraz inne operacje, które przygotowują dane do dalszej analizy i modelowania.

##### Czyszczenie danych

Czyszczenie danych to proces usuwania lub naprawiania błędnych, niepełnych lub niepotrzebnych danych, które mogą wprowadzać szum lub błędy do modelu. Czyszczenie danych może obejmować kilka kroków:

1. **Obsługa brakujących danych:**

Brakujące wartości są powszechnym problemem w danych. Mogą one wystąpić z różnych powodów, takich jak błędy w zbieraniu danych czy problemy techniczne. Sposoby radzenia sobie z brakującymi danymi obejmują:

- **Usuwanie brakujących danych:** Możemy usunąć wiersze lub kolumny, które zawierają brakujące dane, o ile ich liczba jest niewielka i nie wpłynie to negatywnie na model.

- **Imputacja brakujących danych:** Możemy zastąpić brakujące wartości średnią, medianą, modą lub wartościami przewidywanymi przez model.

Przykładowy kod w Pythonie:

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Załóżmy, że mamy DataFrame z brakującymi wartościami
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [5, None, None, 8],
    'C': [9, 10, 11, None]
})

# Usuwanie wierszy z brakującymi wartościami
df_dropna = df.dropna()

# Imputacja brakujących wartości średnią
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("Po usunięciu brakujących wartości:\n", df_dropna)
print("Po imputacji brakujących wartości:\n", df_imputed)
```

2. **Usuwanie duplikatów:**

Duplikaty w danych mogą prowadzić do nadmiernego dopasowania modelu oraz wprowadzać szum do analiz. Należy je zidentyfikować i usunąć.

Przykładowy kod:

```python
# Usuwanie duplikatów
df_cleaned = df_imputed.drop_duplicates()
print("Po usunięciu duplikatów:\n", df_cleaned)
```

3. **Obsługa wartości odstających:**

Wartości odstające (ang. *outliers*) to dane, które znacznie odbiegają od innych wartości w zbiorze danych. Mogą one negatywnie wpływać na modelowanie, zwłaszcza w przypadku modeli wrażliwych na wartości ekstremalne.

- **Detekcja wartości odstających:** Możemy używać metod takich jak Z-score, IQR (Interquartile Range), lub algorytmy dedykowane, np. Isolation Forest.

Przykładowy kod:

```python
import numpy as np

# Detekcja wartości odstających za pomocą Z-score
z_scores = np.abs((df_cleaned - df_cleaned.mean()) / df_cleaned.std())
df_no_outliers = df_cleaned[(z_scores < 3).all(axis=1)]

print("Po usunięciu wartości odstających:\n", df_no_outliers)
```

##### Normalizacja i standaryzacja

Normalizacja i standaryzacja to techniki skalowania danych, które pomagają zapewnić, że cechy są na podobnej skali, co jest istotne dla wielu algorytmów uczenia maszynowego.

1. **Normalizacja:**

Normalizacja polega na przekształceniu wartości cech do przedziału [0, 1] lub [-1, 1], co jest szczególnie przydatne, gdy różne cechy mają różne jednostki miary.

Formuła normalizacji:

$$
x' = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

Przykładowy kod:

```python
from sklearn.preprocessing import MinMaxScaler

# Normalizacja danych
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_no_outliers), columns=df.columns)

print("Po normalizacji:\n", df_normalized)
```

2. **Standaryzacja:**

Standaryzacja polega na przekształceniu wartości cech do rozkładu o średniej 0 i odchyleniu standardowym 1. Jest to szczególnie przydatne dla algorytmów, które zakładają, że dane są normalnie rozłożone.

Formuła standaryzacji:

$$
x' = \frac{x - \mu}{\sigma}
$$

Gdzie:
- $\mu$ to średnia wartość cechy,
- $\sigma$ to odchylenie standardowe cechy.

Przykładowy kod:

```python
from sklearn.preprocessing import StandardScaler

# Standaryzacja danych
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df_no_outliers), columns=df.columns)

print("Po standaryzacji:\n", df_standardized)
```

#### Eksploracyjna analiza danych (EDA)

Eksploracyjna analiza danych (EDA) to proces badania zbioru danych, aby zrozumieć jego strukturę, rozkład, związki między cechami oraz wykryć ewentualne anomalie. EDA jest kluczowym krokiem przed przystąpieniem do modelowania, ponieważ pozwala na uzyskanie wstępnych wniosków i identyfikację potencjalnych problemów.

##### Analiza korelacji

Korelacja mierzy siłę i kierunek związku liniowego między dwiema zmiennymi. W kontekście EDA korelacja jest często używana do identyfikacji cech, które są silnie skorelowane ze zmienną docelową, co może sugerować ich istotność. Równocześnie analiza korelacji pomaga wykryć cechy, które są wysoko skorelowane ze sobą nawzajem, co może prowadzić do redundancji.

1. **Współczynnik korelacji Pearsona:**

Współczynnik korelacji Pearsona to miara liniowej zależności między dwiema zmiennymi:

$$
r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
$$

Gdzie:
- $\bar{x}$ i $\bar{y}$ to średnie wartości zmiennych $x$ i $y$.

Przykładowy kod:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Obliczanie macierzy korelacji
correlation_matrix = df_standardized.corr()

# Wizualizacja macierzy korelacji
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Macierz korelacji')
plt.show()
```

2. **Korelacja Spearmana i Kendalla:**

Gdy mamy do czynienia z danymi nieliniowymi lub rangowymi, możemy użyć współczynnika korelacji Spearmana lub Kendalla, które są miarami monotonicznej zależności między zmiennymi.

Przykładowy kod:

```python
# Korelacja Spearmana
spearman_corr = df_standardized.corr(method='spearman')

# Korelacja Kendalla
kendall_corr = df_standardized.corr(method='kendall')

# Wizualizacja korelacji Spearmana
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm')
plt.title('Korelacja Spearmana')
plt.show()

# Wizualizacja korelacji Kendalla
plt.figure(figsize=(10, 8))
sns.heatmap(kendall_corr, annot=True, cmap='coolwarm')
plt.title('Korelacja Kendalla')
plt.show()
```

##### Wykresy rozrzutu

Wykresy rozrzutu (ang. *scatter plots*) są używane do wizualizacji związku między d

wiema zmiennymi ciągłymi. Pozwalają one na szybkie zidentyfikowanie zależności, trendów, klastrów oraz wartości odstających.

Przykładowy kod:

```python
# Wykres rozrzutu dla dwóch cech
plt.figure(figsize=(8, 6))
plt.scatter(df_standardized['A'], df_standardized['B'])
plt.xlabel('Cechy A')
plt.ylabel('Cechy B')
plt.title('Wykres rozrzutu: Cechy A vs Cechy B')
plt.show()
```

Wykresy rozrzutu są również często używane do badania relacji między cechami a zmienną docelową, co może dostarczyć wskazówek dotyczących nieliniowych zależności, które mogą wymagać specjalnego traktowania w modelu.

#### Implementacja metod selekcji cech

Po przeprowadzeniu preprocessingu i EDA, możemy przystąpić do właściwego procesu selekcji cech. Istnieje wiele metod selekcji cech, które można zastosować, w zależności od natury danych i celu modelu. W tym rozdziale omówimy kilka popularnych metod selekcji cech i porównamy ich wyniki.

##### Przykłady zastosowania różnych metod

1. **Selekcja cech oparta na filtrach:**

Metody filtrów oceniają każdą cechę indywidualnie, niezależnie od modelu, na podstawie pewnych statystyk lub miar.

- **Informacja wzajemna (mutual information):** Mierzy ilość informacji, jaką cecha dostarcza o zmiennej docelowej.

Przykładowy kod:

```python
from sklearn.feature_selection import mutual_info_classif

# Obliczanie informacji wzajemnej dla każdej cechy
mi_scores = mutual_info_classif(df_standardized, y)
mi_scores_series = pd.Series(mi_scores, index=df_standardized.columns)

# Sortowanie i wyświetlanie wyników
mi_scores_sorted = mi_scores_series.sort_values(ascending=False)
print("Informacja wzajemna dla cech:\n", mi_scores_sorted)

# Wizualizacja wyników
mi_scores_sorted.plot(kind='bar', title='Informacja wzajemna dla cech')
plt.show()
```

- **Test chi-kwadrat:** Stosowany głównie do cech kategorialnych, mierzy zależność między cechą a zmienną docelową.

Przykładowy kod:

```python
from sklearn.feature_selection import chi2

# Obliczanie statystyki chi-kwadrat dla każdej cechy
chi2_scores, p_values = chi2(df_standardized, y)

# Konwersja wyników do serii pandas
chi2_scores_series = pd.Series(chi2_scores, index=df_standardized.columns)

# Sortowanie i wyświetlanie wyników
chi2_scores_sorted = chi2_scores_series.sort_values(ascending=False)
print("Chi-kwadrat dla cech:\n", chi2_scores_sorted)

# Wizualizacja wyników
chi2_scores_sorted.plot(kind='bar', title='Chi-kwadrat dla cech')
plt.show()
```

2. **Selekcja cech oparta na osadzaniu (embedded):**

Metody osadzania integrują selekcję cech z procesem trenowania modelu.

- **LASSO (Least Absolute Shrinkage and Selection Operator):** Regularizacja L1, która może zerować współczynniki wag cech, skutecznie eliminując nieistotne cechy.

Przykładowy kod:

```python
from sklearn.linear_model import Lasso

# Tworzenie modelu LASSO
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(df_standardized, y)

# Wagi cech
lasso_coef = pd.Series(lasso_model.coef_, index=df_standardized.columns)

# Wyświetlanie i wizualizacja wyników
lasso_coef_sorted = lasso_coef.sort_values(ascending=False)
print("Wagi cech (LASSO):\n", lasso_coef_sorted)

lasso_coef_sorted.plot(kind='bar', title='Wagi cech (LASSO)')
plt.show()
```

- **Drzewa decyzyjne i lasy losowe:** Ocena cech na podstawie ich znaczenia w modelu drzewa decyzyjnego lub lasów losowych.

Przykładowy kod:

```python
from sklearn.ensemble import RandomForestClassifier

# Tworzenie modelu lasów losowych
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(df_standardized, y)

# Znaczenie cech
rf_importances = pd.Series(rf_model.feature_importances_, index=df_standardized.columns)

# Wyświetlanie i wizualizacja wyników
rf_importances_sorted = rf_importances.sort_values(ascending=False)
print("Znaczenie cech (Lasy losowe):\n", rf_importances_sorted)

rf_importances_sorted.plot(kind='bar', title='Znaczenie cech (Lasy losowe)')
plt.show()
```

3. **Selekcja cech oparta na przeszukiwaniu (wrapper):**

Metody przeszukiwania oceniają różne kombinacje cech, aby znaleźć optymalny podzbiór cech dla danego modelu.

- **Rekurencyjna Eliminacja Cech (RFE):** Metoda iteracyjna, która stopniowo eliminuje najmniej istotne cechy.

Przykładowy kod:

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Tworzenie modelu bazowego
logreg = LogisticRegression(max_iter=1000)

# RFE
rfe = RFE(logreg, n_features_to_select=5)
rfe.fit(df_standardized, y)

# Wyświetlanie wyników
rfe_ranking = pd.Series(rfe.ranking_, index=df_standardized.columns)
print("Ranking cech (RFE):\n", rfe_ranking)

# Wizualizacja wyników
rfe_ranking_sorted = rfe_ranking.sort_values()
rfe_ranking_sorted.plot(kind='bar', title='Ranking cech (RFE)')
plt.show()
```

##### Porównanie wyników różnych metod

W praktyce, różne metody selekcji cech mogą dawać różne wyniki. Porównanie wyników różnych metod może dostarczyć cennych informacji na temat tego, które cechy są najbardziej istotne dla danego problemu.

Przykładowo, można porównać zestawy cech wybrane przez metodę RFE, LASSO oraz informacje wzajemną, aby zobaczyć, które cechy są wybierane najczęściej:

```python
# Przykładowe zestawy cech wybrane przez różne metody
rfe_selected = rfe.support_
lasso_selected = lasso_coef_sorted.index[lasso_coef_sorted != 0]
mi_selected = mi_scores_sorted.index[mi_scores_sorted > np.median(mi_scores_sorted)]

# Porównanie wybranych cech
selected_features = pd.DataFrame({
    'RFE': rfe_selected,
    'LASSO': df_standardized.columns.isin(lasso_selected),
    'MI': df_standardized.columns.isin(mi_selected)
})

print("Porównanie wybranych cech:\n", selected_features)
```

Porównanie wyników może prowadzić do konsensusu lub wskazać na konieczność dalszej analizy. Warto również pamiętać, że w praktyce dobór cech często wymaga iteracyjnego podejścia i testowania różnych konfiguracji na podstawie wydajności modelu na zbiorze walidacyjnym.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone
