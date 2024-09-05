## 6. **Przypadki Użycia**

W kontekście uczenia maszynowego, wybór cech to kluczowy krok, który może zdecydować o sukcesie lub porażce modelu. Aby zilustrować, jak różne techniki selekcji cech mogą wpływać na jakość modelu, omówimy kilka przypadków użycia opartych na rzeczywistych projektach. Porównamy efektywność modeli z różnymi zestawami cech i przeanalizujemy wyniki, wyciągając wnioski dotyczące praktycznych aspektów selekcji cech.

#### Studia przypadków z rzeczywistych projektów

##### Przypadek 1: Prognozowanie cen nieruchomości

Wybór cech jest kluczowy w modelach prognozowania cen nieruchomości, gdzie dane mogą zawierać różnorodne informacje, od lokalizacji i wielkości nieruchomości po bardziej szczegółowe dane, takie jak odległość od szkół, dostęp do komunikacji miejskiej, czy stan techniczny budynku.

**Cel projektu:**
Zbudowanie modelu, który dokładnie przewiduje ceny nieruchomości na podstawie dostępnych danych.

**Zbiór danych:**
Dane zebrane z agencji nieruchomości, zawierające informacje o lokalizacji, powierzchni, liczbie pokoi, roku budowy, odległości od centrów handlowych, szkół, dostępności komunikacji miejskiej, a także o standardzie wykończenia.

**Kroki realizacji:**

1. **Wstępna analiza danych:**

   Przeprowadzono wstępną analizę danych, aby zidentyfikować brakujące wartości, wartości odstające oraz silne korelacje między cechami.

   ```python
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt

   # Załadowanie danych
   df = pd.read_csv("house_prices.csv")

   # Sprawdzenie brakujących danych
   print(df.isnull().sum())

   # Korelacja
   correlation_matrix = df.corr()
   sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
   plt.title('Macierz korelacji')
   plt.show()
   ```

2. **Selekcja cech przy użyciu LASSO:**

   Regularyzacja LASSO została zastosowana w celu wyeliminowania nieistotnych cech. Model LASSO pozwala na zmniejszenie współczynników niektórych cech do zera, co efektywnie eliminuje je z modelu.

   ```python
   from sklearn.linear_model import Lasso
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_squared_error

   # Podział danych
   X = df.drop("price", axis=1)
   y = df["price"]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # LASSO
   lasso = Lasso(alpha=0.1)
   lasso.fit(X_train, y_train)

   # Wagi cech
   print("Wagi cech (LASSO):", lasso.coef_)

   # Prognozowanie
   y_pred = lasso.predict(X_test)
   print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
   ```

**Wynik:**
- Model LASSO pomógł zidentyfikować kluczowe cechy, takie jak lokalizacja, powierzchnia i stan techniczny nieruchomości, jednocześnie eliminując mniej istotne cechy, jak liczba szafek w kuchni czy odległość do najbliższego sklepu.

**Wnioski:**
- Zastosowanie LASSO pozwoliło na uproszczenie modelu bez utraty jego dokładności, co przyspieszyło trening i poprawiło interpretowalność wyników.

##### Przypadek 2: Klasyfikacja klientów w banku

Banki często stosują modele uczenia maszynowego do klasyfikacji klientów pod kątem ryzyka kredytowego. Wybór odpowiednich cech jest tutaj kluczowy, ponieważ dane mogą zawierać wiele informacji, które niekoniecznie wpływają na ryzyko niespłacenia kredytu.

**Cel projektu:**
Zbudowanie modelu klasyfikacyjnego, który przewiduje, czy klient spłaci kredyt, czy nie.

**Zbiór danych:**
Dane obejmują informacje demograficzne klientów, historię kredytową, wysokość dochodów, liczbę dzieci, stan cywilny, oraz dane dotyczące wcześniejszych kredytów.

**Kroki realizacji:**

1. **Wstępna analiza danych:**

   Dane zostały przetworzone w celu zidentyfikowania potencjalnie problematycznych cech, takich jak zmienne o niskiej wariancji, które mogą nie wnosić wartości do modelu.

   ```python
   # Sprawdzenie wariancji cech
   variances = X.var()
   print("Cechy o niskiej wariancji:\n", variances[variances < 0.01])
   ```

2. **Selekcja cech przy użyciu metod drzewiastych:**

   Model lasów losowych został zastosowany w celu oceny ważności cech, co pozwoliło na zidentyfikowanie tych, które mają największy wpływ na decyzję o przyznaniu kredytu.

   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score

   # Model lasów losowych
   rf = RandomForestClassifier(n_estimators=100, random_state=42)
   rf.fit(X_train, y_train)

   # Znaczenie cech
   feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
   feature_importances.sort_values().plot(kind='barh', title='Znaczenie cech')
   plt.show()

   # Prognozowanie
   y_pred = rf.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, y_pred))
   ```

**Wynik:**
- Ważne cechy obejmowały historię kredytową, wysokość dochodów i rodzaj zatrudnienia, podczas gdy cechy takie jak liczba dzieci czy stan cywilny okazały się mniej istotne.

**Wnioski:**
- Modele drzewiaste pozwalają na automatyczną selekcję najważniejszych cech, co jest szczególnie przydatne w projektach z dużą liczbą potencjalnych zmiennych.

##### Przypadek 3: Rozpoznawanie obrazów

Rozpoznawanie obrazów to jeden z obszarów, w którym sieci neuronowe, a zwłaszcza konwolucyjne sieci neuronowe (CNN), mają ogromne zastosowanie. Wybór cech w tym kontekście może oznaczać selekcję odpowiednich filtrów, warstw lub technik augmentacji danych.

**Cel projektu:**
Zbudowanie modelu, który klasyfikuje obrazy na podstawie zestawu danych zawierającego obrazy psów i kotów.

**Zbiór danych:**
Zestaw danych obrazowych zawierający tysiące zdjęć psów i kotów.

**Kroki realizacji:**

1. **Wstępna analiza danych:**

   Dane obrazowe zostały poddane procesowi eksploracji w celu zrozumienia ich struktury oraz identyfikacji potencjalnych problemów, takich jak nierównowaga klas.

   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator

   # Wizualizacja kilku obrazów
   datagen = ImageDataGenerator(rescale=1./255)
   train_generator = datagen.flow_from_directory(
       'data/train', target_size=(150, 150), batch_size=20, class_mode='binary')

   import matplotlib.pyplot as plt
   for i in range(9):
       plt.subplot(330 + 1 + i)
       plt.imshow(train_generator[0][0][i])
   plt.show()
   ```

2. **Selekcja cech przez sieć konwolucyjną (CNN):**

   CNN zostały zaprojektowane w taki sposób, aby same mogły uczyć się odpowiednich cech z obrazów. Można jednak zastosować techniki takie jak regularyzacja L2 czy dropout, aby poprawić generalizację modelu.

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
   from tensorflow.keras.optimizers import Adam

   # Budowa modelu CNN
   model = Sequential([
       Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
       MaxPooling2D(2, 2),
       Conv2D(64, (3, 3), activation='relu'),
       MaxPooling2D(2, 2),
       Conv2D(128, (3, 3), activation='relu'),
       MaxPooling2D(2, 2),
       Flatten(),
       Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
       Dropout(0.5),
       Dense(1, activation='sigmoid')
   ])

   ## Kompilowanie modelu
   model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

   # Trenowanie modelu
   history = model.fit(train_generator, epochs=10, validation_data=val_generator)
   ```

**Wynik:**
- CNN nauczyła się rozpoznawać istotne cechy, takie jak kształty, tekstury, i wzory, które pozwalają na odróżnienie psów od kotów.

**Wnioski:**
- Wybór cech w sieciach neuronowych może być częściowo automatyczny, ale zastosowanie technik regularyzacyjnych takich jak L2 czy dropout jest kluczowe, aby model nie nauczył się nieistotnych detali (overfitting).

#### Porównanie efektywności modeli z różnymi zestawami cech

Aby lepiej zrozumieć, jak wybór cech wpływa na efektywność modeli, przeprowadziliśmy porównanie modeli trenowanych na różnych zestawach cech.

**Przykład: Model klasyfikacji z danymi dotyczącymi zdrowia**

**Cel projektu:**
Porównanie efektywności modeli klasyfikacyjnych przewidujących ryzyko choroby serca przy użyciu różnych zestawów cech.

**Zbiór danych:**
Zestaw danych medycznych zawierający informacje takie jak wiek, płeć, poziom cholesterolu, ciśnienie krwi, palenie papierosów, historia chorób rodzinnych, itp.

**Kroki realizacji:**

1. **Pełny zestaw cech:**

   Trening modelu na pełnym zestawie cech, bez selekcji.

   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import roc_auc_score

   # Pełny zestaw cech
   model_full = LogisticRegression(max_iter=1000)
   model_full.fit(X_train, y_train)

   # Prognozowanie i ocena
   y_pred_full = model_full.predict(X_test)
   auc_full = roc_auc_score(y_test, y_pred_full)
   print("AUC na pełnym zestawie cech:", auc_full)
   ```

2. **Zredukowany zestaw cech po selekcji:**

   Trening modelu na zestawie cech wybranym przy użyciu LASSO.

   ```python
   # Selekcja cech przy użyciu LASSO
   lasso = Lasso(alpha=0.1)
   lasso.fit(X_train, y_train)

   # Tylko wybrane cechy
   X_train_selected = X_train.loc[:, lasso.coef_ != 0]
   X_test_selected = X_test.loc[:, lasso.coef_ != 0]

   # Trening modelu na wybranych cechach
   model_selected = LogisticRegression(max_iter=1000)
   model_selected.fit(X_train_selected, y_train)

   # Prognozowanie i ocena
   y_pred_selected = model_selected.predict(X_test_selected)
   auc_selected = roc_auc_score(y_test, y_pred_selected)
   print("AUC na wybranych cechach:", auc_selected)
   ```

**Wynik:**
- Model zredukowany na podstawie LASSO osiągnął podobny wynik AUC jak model z pełnym zestawem cech, ale miał mniej parametrów, co mogło zmniejszyć ryzyko nadmiernego dopasowania.

**Wnioski:**
- Selekcja cech może prowadzić do uproszczenia modelu bez utraty jego wydajności, co jest szczególnie cenne w zastosowaniach praktycznych, gdzie interpretowalność i szybkość są równie ważne jak dokładność.

#### Analiza wyników i wnioski

Na podstawie przedstawionych studiów przypadków i wyników porównania można wyciągnąć kilka kluczowych wniosków:

1. **Wpływ selekcji cech na interpretowalność modelu:**
   - Wybór odpowiednich cech znacząco poprawia interpretowalność modelu. Zamiast opierać decyzje na wszystkich dostępnych cechach, co może prowadzić do złożonych i niejasnych modeli, selekcja cech pozwala skupić się na tych zmiennych, które mają rzeczywisty wpływ na wynik.

2. **Efektywność i szybkość modelowania:**
   - Zmniejszenie liczby cech zazwyczaj prowadzi do szybszego treningu modelu. Jest to szczególnie ważne w kontekście dużych zbiorów danych, gdzie każda dodatkowa cecha zwiększa czas obliczeń oraz zasoby wymagane do przetwarzania.

3. **Nadmierne dopasowanie (overfitting):**
   - Eliminacja nieistotnych cech może pomóc w redukcji nadmiernego dopasowania. Modele o mniejszej liczbie cech są często bardziej generalizowalne i lepiej radzą sobie na danych testowych.

4. **Znaczenie metody selekcji cech:**
   - Różne metody selekcji cech mają różne zalety i wady. Na przykład, LASSO jest skuteczne w modelach liniowych, ale w przypadku bardziej złożonych relacji między zmiennymi, takie jak te obsługiwane przez modele drzewiaste, metody oparte na ważności cech mogą dawać lepsze wyniki.

5. **Praktyczne zastosowanie selekcji cech:**
   - W praktyce warto łączyć różne metody selekcji cech, aby uzyskać jak najlepszy zestaw cech. Na przykład, można zacząć od filtrów, aby usunąć ewidentnie nieistotne cechy, a następnie zastosować metody osadzania, takie jak LASSO lub modele drzewiaste, aby dokonać końcowej selekcji.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone
