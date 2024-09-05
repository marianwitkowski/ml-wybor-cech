## 7. **Wyzwania i Problemy w Wyborze Cech**

Wybór cech jest jednym z najważniejszych etapów w procesie budowy modeli uczenia maszynowego, ale jest także źródłem wielu wyzwań i problemów. Efektywna selekcja cech może znacząco poprawić wydajność modelu, ale złożoność tego procesu nie powinna być niedoceniana. W tej sekcji omówimy trzy główne wyzwania związane z wyborem cech: przekleństwo wymiarowości, problem nadmiernego dopasowania oraz radzenie sobie z cechami redundantnymi i nieistotnymi.

#### Przekleństwo wymiarowości

##### Wprowadzenie

Przekleństwo wymiarowości (ang. *curse of dimensionality*) odnosi się do problemów, które pojawiają się, gdy liczba cech (wymiarów) w zbiorze danych jest bardzo duża w porównaniu do liczby obserwacji. Wysoka wymiarowość może prowadzić do rozmaitych problemów, takich jak trudności w modelowaniu, zwiększona złożoność obliczeniowa oraz większe ryzyko nadmiernego dopasowania.

##### Problemy związane z przekleństwem wymiarowości

1. **Rzadka przestrzeń cech:**
   - W miarę wzrostu liczby cech, przestrzeń cech staje się coraz bardziej rzadka, co oznacza, że obserwacje stają się od siebie coraz bardziej oddalone. Wysoka wymiarowość powoduje, że dane stają się rozproszone, co może utrudniać modelowi identyfikację wzorców.

2. **Zwiększona złożoność obliczeniowa:**
   - Większa liczba cech zwiększa liczbę operacji obliczeniowych potrzebnych do analizy danych. Może to prowadzić do wydłużenia czasu treningu oraz większych wymagań pamięciowych.

3. **Wysokie ryzyko nadmiernego dopasowania:**
   - Z większą liczbą cech model może łatwo nauczyć się szumu w danych zamiast prawdziwych wzorców, co prowadzi do nadmiernego dopasowania.

##### Techniki redukcji wymiarowości

1. **Analiza głównych składowych (PCA):**

   PCA to technika redukcji wymiarowości, która przekształca cechy do nowej przestrzeni, w której nowe cechy (główne składowe) są nieskorelowane i maksymalizują wariancję danych. PCA zachowuje jak najwięcej informacji, jednocześnie redukując wymiarowość.

   Wzór na transformację PCA:

   $Z = XW$

   Gdzie:
   - $Z$ to macierz nowych cech (głównych składowych),
   - $X$ to oryginalna macierz cech,
   - $W$ to macierz wektorów własnych odpowiadających największym wartościom własnym.

   Przykładowy kod w Pythonie:

   ```python
   from sklearn.decomposition import PCA
   import matplotlib.pyplot as plt

   # Redukcja wymiarowości przy użyciu PCA
   pca = PCA(n_components=2)  # Redukcja do 2 wymiarów dla wizualizacji
   X_pca = pca.fit_transform(X)

   # Wizualizacja wyników PCA
   plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
   plt.xlabel('Główna składowa 1')
   plt.ylabel('Główna składowa 2')
   plt.title('Wynik PCA')
   plt.show()
   ```

3. **Selekcja cech oparta na filtrach:**

   Filtry oceniają każdą cechę niezależnie i wybierają te, które są najbardziej istotne na podstawie miar takich jak korelacja, informacja wzajemna czy test chi-kwadrat. Filtry mogą być stosowane jako wstępna metoda redukcji wymiarowości przed zastosowaniem bardziej zaawansowanych technik.

   Przykładowy kod w Pythonie:

   ```python
   from sklearn.feature_selection import SelectKBest, chi2

   # Selekcja najlepszych cech przy użyciu testu chi-kwadrat
   selector = SelectKBest(chi2, k=5)
   X_new = selector.fit_transform(X, y)

   print("Wybrane cechy:\n", X_new)
   ```

##### Przykład zastosowania

Załóżmy, że pracujemy z danymi genetycznymi, gdzie każda cecha reprezentuje ekspresję konkretnego genu. Liczba genów (cech) może być ogromna, podczas gdy liczba próbek (obserwacji) jest ograniczona. W takim przypadku, zastosowanie PCA może pomóc w redukcji wymiarowości, jednocześnie zachowując najważniejsze informacje, co pozwala na efektywniejsze trenowanie modelu.

#### Problem nadmiernego dopasowania (overfitting)

##### Wprowadzenie

Nadmierne dopasowanie (ang. *overfitting*) występuje, gdy model zbyt dobrze dopasowuje się do danych treningowych, ale nie generalizuje dobrze na nowych danych. Może to być spowodowane zbyt dużą liczbą cech w stosunku do liczby próbek, co prowadzi do nauki szumu w danych zamiast rzeczywistych wzorców.

##### Objawy nadmiernego dopasowania

1. **Wysoka dokładność na danych treningowych, niska na testowych:**
   - Model osiąga bardzo dobre wyniki na danych treningowych, ale jego wydajność na zbiorze testowym jest znacznie gorsza.

2. **Zbyt skomplikowany model:**
   - Model z dużą liczbą cech i złożoną strukturą może zbyt dokładnie odwzorowywać dane treningowe, zamiast uogólniać wzorce.

##### Metody radzenia sobie z nadmiernym dopasowaniem

1. **Regularyzacja:**

   Regularyzacja dodaje karę do funkcji straty za zbyt duże wartości wag cech, co pomaga w uproszczeniu modelu i zmniejszeniu nadmiernego dopasowania.

   - **L2 (Ridge) regularization:** Dodaje karę za sumę kwadratów wag.

     
     $\text{Funkcja straty} = \text{RSS} + \lambda \sum_{i=1}^{n} w_i^2$

   - **L1 (LASSO) regularization:** Dodaje karę za sumę wartości bezwzględnych wag, co może prowadzić do wyzerowania niektórych wag.

     $\text{Funkcja straty} = \text{RSS} + \lambda \sum_{i=1}^{n} |w_i|$

   Przykładowy kod w Pythonie:

   ```python
   from sklearn.linear_model import Ridge, Lasso

   # Ridge regularization
   ridge = Ridge(alpha=1.0)
   ridge.fit(X_train, y_train)
   print("Ridge - Współczynniki:", ridge.coef_)

   # LASSO regularization
   lasso = Lasso(alpha=0.1)
   lasso.fit(X_train, y_train)
   print("LASSO - Współczynniki:", lasso.coef_)
   ```

2. **Walidacja krzyżowa (Cross-validation):**

   Walidacja krzyżowa to technika oceny modelu, w której dane są dzielone na kilka podzbiorów (tzw. foldów). Model jest trenowany na kilku podzbiorach, a testowany na pozostałych, co pozwala na ocenę jego zdolności do generalizacji.

   Przykładowy kod w Pythonie:

   ```python
   from sklearn.model_selection import cross_val_score

   # Model liniowy z walidacją krzyżową
   model = Ridge(alpha=1.0)
   scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
   print("Średni błąd kwadratowy w walidacji krzyżowej:", -scores.mean())
   ```

3. **Dropout (w sieciach neuronowych):**

   Dropout to technika regularyzacyjna stosowana w sieciach neuronowych, która losowo wyłącza pewien procent neuronów podczas treningu, co pomaga w zapobieganiu nadmiernemu dopasowaniu.

   Przykładowy kod w Pythonie:

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense, Dropout

   # Sieć neuronowa z Dropout
   model = Sequential([
       Dense(128, activation='relu', input_shape=(X.shape[1],)),
       Dropout(0.5),
       Dense(128, activation='relu'),
       Dropout(0.5),
       Dense(1, activation='sigmoid')
   ])

   # Kompilowanie modelu
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

   # Trenowanie modelu
   model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
   ```

##### Przykład zastosowania

Rozważmy przykład klasyfikacji wniosków kredytowych na podstawie danych zawierających cechy takie jak wiek, dochód, liczba kart kredytowych, historia kredytowa itp. W takim przypadku, jeśli dodamy zbyt wiele nieistotnych cech (np. liczba dzieci czy stan cywilny), model może dopasować się do tych cech, zamiast skupić się na kluczowych informacjach. Regularyzacja L1 lub L2 może pomóc w zmniejszeniu wagi mniej istotnych cech i poprawie zdolności modelu do generalizacji.

#### Radzenie sobie z cechami redundantnymi i nieistotnymi

##### Wprowadzenie

Cechy redundantne to te, które dostarczają tej samej informacji co inne cechy w zbiorze danych, podczas gdy cechy nieistotne to te, które nie mają znaczącego wpływu na zmienną docelową. Obecność takich cech w modelu może prowadzić do nadmiernego dopasowania, zwiększenia złożoności modelu oraz wydłużenia czasu treningu.

##### Problemy związane z cechami redundantnymi i nieistotnymi

1. **Zwiększona złożoność modelu:**
   - Dodanie cech, które nie wnoszą wartości, zwiększa liczbę parametrów modelu, co może prowadzić do jego komplikacji i trudności w interpretacji.

2. **Nadmierne dopasowanie:**
   - Model może "nauczyć się" nieistotnych cech, co prowadzi do nadmiernego dopasowania i gorszej wydajności na nowych danych.

3. **Zwiększone koszty obliczeniowe:**
   - Więcej cech oznacza większe wymagania obliczeniowe podczas trenowania i predykcji modelu, co może być problematyczne w przypadku bardzo dużych zbiorów danych.

##### Techniki radzenia sobie z cechami redundantnymi i nieistotnymi

1. **Analiza korelacji:**

   Analiza korelacji pozwala zidentyfikować cechy, które są silnie skorelowane ze sobą, co może sugerować, że niektóre z nich są redundantne. Takie cechy mogą być usunięte lub zredukowane.

   Przykładowy kod w Pythonie:

   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   # Obliczanie macierzy korelacji
   correlation_matrix = X.corr()

   # Wizualizacja korelacji
   plt.figure(figsize=(12, 10))
   sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
   plt.title('Macierz korelacji')
   plt.show()

   # Usuwanie cech o wysokiej korelacji (np. r > 0.9)
   upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
   redundant_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
   X_reduced = X.drop(columns=redundant_features)
   ```

2. **Selekcja cech na podstawie ważności (Feature importance):**

   W modelach takich jak lasy losowe, znaczenie cech może być używane do zidentyfikowania i usunięcia cech, które mają niewielki wpływ na wynik modelu.

   Przykładowy kod w Pythonie:

   ```python
   from sklearn.ensemble import RandomForestClassifier

   # Trening modelu lasów losowych
   rf = RandomForestClassifier(n_estimators=100)
   rf.fit(X_train, y_train)

   # Znaczenie cech
   importances = pd.Series(rf.feature_importances_, index=X.columns)

   # Usuwanie nieistotnych cech (np. importance < 0.01)
   X_selected = X.loc[:, importances > 0.01]
   ```

3. **Algorytmy osadzania (Embedded methods):**

   Metody takie jak LASSO, które integrują selekcję cech z procesem trenowania modelu, mogą automatycznie eliminować nieistotne cechy.

   Przykładowy kod w Pythonie:

   ```python
   from sklearn.linear_model import Lasso

   # LASSO
   lasso = Lasso(alpha=0.1)
   lasso.fit(X_train, y_train)

   # Wybór cech
   X_selected_lasso = X_train.loc[:, lasso.coef_ != 0]
   ```

##### Przykład zastosowania

Przy budowie modelu prognozowania cen nieruchomości, może okazać się, że niektóre cechy, takie jak liczba okien w salonie czy kolor dachu, są mało istotne. Usunięcie takich cech może uprościć model i poprawić jego zdolność do generalizacji.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone
