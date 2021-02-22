#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importovanje potrebnih biblioteka
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn
from scipy import stats
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#citanje csv fajla sa separatorom ;
bank = pd.read_csv('bank-additional.csv',sep=';')


# In[3]:


#kada izvrsimo funkciju info() dobijamo tipove vrednosti za svaku kolonu. Vidimo da najvise ima kolona object tipa i float64
bank.info()
#funkcija describe vraca ukupan broj instanci, aritmeticku sredinu, standardnu devijaciju, minimalnu, maksimalnu vrednost, prvi i treci kvartil i medijanu u obliku tabale
bank.describe()


# In[4]:


#za kategoricke promenjive vidimo da sve osim nr.employed i y nemaju NA vrednosti. Sve kategoricke varijable imaju maksimalno 12 jedinstvenih vrednosti sto je dobra stvar 
#(ukoliko imamo puno unique vrednosti u nekoj promenjivi tu promenjivu bismo morali da izbacimo). Ovde vidimo i izlaznu promenjivu y koja ima 2 unique vrednosti.
bank.describe(include=[np.object])


# In[5]:


#funkcijom isna().sum() vidimo ukupan broj nepoznatih vrednosti po kolonama (npr. campaign ih ima 5, duration 3 a age 0)
bank.isna().sum()


# In[6]:


#vidimo da ovde bank['nr.employed'] ima jednu vrednost 'no' sto nije dobro jer nr.employed treba biti numericki tip tako da cemo kastovati to_numeric sa error type 'coerce'.
#Error 'coerce' znaci da sve vrednosti koje ne mogu biti parsirane, u ovom slucaju u numericke, biti proglasane za NA pa cemo tako dobiti jos jednu NA vrednost za atribut 'nr.employed'.
sum(bank['nr.employed']=='no')


# In[7]:


#ovom funkcijom sredjujemo da se kolona nr.employed kastuje iz object-a u float64
bank['nr.employed']=bank['nr.employed'].apply(pd.to_numeric, errors='coerce')
bank.isna().sum()


# In[8]:


#preslikivanje izlaza u binarni formu
bank['y'] = bank['y'].map({'no':0, 'yes':1})


# In[9]:


#sada radimo za sve vrednosti koje imaju bar jednu NA vrednost test normalnosti. 
#Ako ima normalnu raspodelu NA vrednost menjamo sa aritmetickom sredinom (mean) a ako nema onda medijanom.
#sve p vrednosti preko 0.5 znaci normalnu raspodelu dok ispod 0.5 znaci da atribut nema normalnu rasporedelu


# In[10]:


stats.normaltest(bank['nr.employed'],nan_policy='omit')


# In[11]:


stats.normaltest(bank['euribor3m'],nan_policy='omit')


# In[12]:


stats.normaltest(bank['cons.conf.idx'],nan_policy='omit')


# In[13]:


stats.normaltest(bank['cons.price.idx'],nan_policy='omit')


# In[14]:


stats.normaltest(bank['emp.var.rate'],nan_policy='omit')


# In[15]:


stats.normaltest(bank['campaign'],nan_policy='omit')


# In[16]:


stats.normaltest(bank['duration'],nan_policy='omit')


# In[17]:


#posto ni za jednu promenjivu ne mozemo sa velikom znacajnoscu da kazemo da ima normalnu rasporedelu, onda cemo sve NA vrednosti zameni sa medijanom
bank = bank.fillna(bank.median())
bank.isna().sum()


# In[18]:


#numericka korealaciona matrica
#Sa korelacione matrice vidimo da najbolju korelaciju imamju eurobor3m i nr.employed koja iznosi cak 0.89, emp.var.rate i nr.employed 0.76 a najbolju negativnu korelaciju previous i pdays od -0.58.
#Mozemo da razmatrimo da izbacimo promenjivu emp.var.vare jer ona ima jaku korelaciju sa nr.employed i sa cons.price.inx pa dolazi do redudanse podataka.
bank.corr()
#korelaciona matrica sa brojevima i bojama u zavisnosti od koeficijenta korelacije
corr = bank.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[19]:


bank = bank.drop('emp.var.rate', 1)


# In[20]:


bank.info()


# In[21]:


sns.heatmap(corr)


# In[22]:


bank.isna().sum()


# In[23]:


bank.isnull().sum()


# In[24]:


#prikazivanje linearnosti promenjivih. Linearnost izmedju promenjivih pokazuje medjuzavisnost izmedju dve promenjive.
#Za predvidjanja nije dobra visoka linearnost izmedju ulaznih paramatera jer dolazi do redudanse podataka.
from pandas.plotting import scatter_matrix
scatter_matrix(bank, figsize=(20,10))


# In[25]:


#iscrtavanje raspodele podatakata numerickog tipa. Algoritmi masinskog ucenja najbolje rezultate pokazuju sa podaci koji imaju normalnu raspodelu,
#medjutim ovde sa slike vidimo da to nije tako (kasnije je radjen i Pearson-ov test normlanosti koji je potvrdio da nemaju normalnu rasporedelu).
bank.hist(figsize=(20,20)) 


# In[26]:


#pravljenje kategorickih promenjivih
categorical = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for i in categorical:
    bank[i] = bank[i].astype('category')  


# In[27]:


#crtanje raspodele kateogirckih promenjivih. Za svaku klasu kategorickih podataka, vidimo broj instanci.
#Za predvidjanja veliki broj razlictih klasa u jednoj instanci nije dobar ali ovde to nije slucaj jer je maksmilan broj 12.
categorical = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for i in categorical:
    g = sn.catplot(x=i, kind="count", palette="ch:.25",data=bank);
    g.fig.set_size_inches(15,7)


# In[28]:


#popunjavanje nepoznatih vrednosti medijanom
bank = bank.fillna(bank.median())
bank.isna().sum()


# categorical = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
# for i in categorical:
#     sn.catplot(x=i, kind="count", palette="ch:.25",figsize=(8, 6), data=bank);

# In[29]:


#pravljenje ulaza i izlaza modela
X = bank.iloc[:,:-1]
y = bank.iloc[:,-1]


# In[30]:


#kreiranje dummies promenjivih od kategorickih varijabli
pom = X.select_dtypes(include=['int64','float64'])
for i in categorical:
    X[i] = X[i].astype('category')
    dummy = pd.get_dummies(X[i])
    pom = pd.concat([pom,dummy],axis=1)
X = pom
X.info()


# In[31]:


#funkcija za prikaz ocene tacnosti modela
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score

def print_performance(y_parametar, y_hat):
    print(f'Accuracy: {accuracy_score(y_parametar, y_hat)}')
    print(f'Precision: {precision_score(y_parametar, y_hat)}')
    print(f'Recall: {recall_score(y_parametar, y_hat)}')
    print(f'F1: {f1_score(y_parametar, y_hat)}')


# In[32]:


#funkcija za ocenu modela samo na trening setu
#accurarcy se dobija kada se broj tacnih predvidjenih podeli sa brojem ukupnih predvidjanja (kolicnik zbira na glavnoj 
#dijagonali matrice konfuzije i zbira svih elemenata matrice konfuzije). On pokazuje koliko je nas model tacan.
#precison je kolicnik zbira True pozivno predvidjenih sa ukupno True predvidjenim. Odgovora na pitanje koliki je procenatac tacno predvidjenih ustvari tacan
#recall odgovora na pitanje koliko je stvarnih true positive predvidjeno modelom. Kod nekih sistema, kao sto je na primer otkrivanje bolesti, je potrebno
#da ovaj model ima sto vecu vrednost.
#precision i recall su suprotne vrednosti, kada jedna opada druga raste i obrnuto
#f1 oznacava meru izmedju precision i recall.
from sklearn.metrics import confusion_matrix
def ocena_trening(algoritam,X_parametar,y_parametar):
    algoritam.fit(X_parametar,y_parametar)
    y_hat = algoritam.predict(X_parametar)
    print_performance(y_parametar,y_hat)
    print(confusion_matrix(y_true=y_parametar,y_pred=algoritam.predict(X_parametar)))


# In[33]:


#Vidimo da se prefornse kod Logisticke regresije odlicne
from sklearn.linear_model import LogisticRegression
ocena_trening(LogisticRegression(),X,y)


# In[34]:


from sklearn.naive_bayes import GaussianNB
ocena_trening(GaussianNB(),X,y)


# In[35]:


from sklearn.neighbors import KNeighborsClassifier
ocena_trening(KNeighborsClassifier(),X,y)


# In[36]:


#Perfomanse kod drveta odlucivanja su odlicne sto je posledice overffiting-a
from sklearn.tree import DecisionTreeClassifier
ocena_trening(DecisionTreeClassifier(),X,y)


# In[37]:


#funkcija koja deli podatke na trening i test set u proporciji 70:30, zatim kreira model na osnovu treninga, predvidja rezultate test podataka i upordjuje
#ovakve izlaze sa stvarnim izlazima u matrici konfuzije
from sklearn.model_selection import train_test_split
def ocena_test(algoritam,X_parametar,y_parametar):
    X_train, X_test,y_train, y_test = train_test_split(X_parametar,y_parametar,test_size=0.3,random_state=2020)
    algoritam.fit(X_train,y_train)
    y_hat = algoritam.predict(X_test)
    print_performance(y_test,y_hat)
    confusion_matrix(y_true=y_test,y_pred=algoritam.predict(X_test))
    print(confusion_matrix(y_true=y_test,y_pred=algoritam.predict(X_test)))


# In[38]:


#ovde kod Logisticke regresije vidimo da model daje bolje rezultate se podeli na trening i test skup
ocena_test(LogisticRegression(),X,y)


# In[39]:


#ovde su performanse modela opale u odnosu na model sa samo trening setom
ocena_test(GaussianNB(),X,y)


# In[40]:


#ovde su performanse modela opale u odnosu na model sa samo trening setom
ocena_test(KNeighborsClassifier(),X,y)


# In[41]:


#ovde su performanse modela znatno opale u odnosu na model sa samo trening setom. U trening setu model je u potpunosti bio tacan medjutim to je posledica overffiting-a
ocena_test(DecisionTreeClassifier(),X,y)


# In[42]:


#funkcija koja radi cross-validaciju podelom na 10 delova, za svaki tako dobijen model prikazuje tacnost i na kraju prikazuje tacnost citave cross-validatcije
from sklearn.model_selection import KFold
def cross_validation(algoritam,X_parametar,y_parametar):
    folds = KFold(n_splits=10)
    results = []
    for train_index, test_index in folds.split(X_parametar):
        X_train, y_train  = X_parametar.loc[train_index, :], y_parametar[train_index]
        X_test, y_test = X_parametar.loc[test_index], y_parametar[test_index]
    
        algoritam.fit(X_train, y_train)
    
        y_hat = algoritam.predict(X_test)
        results.append(accuracy_score(y_test, y_hat))
        print(f'Accuracy: {accuracy_score(y_test, y_hat)}')
    print(f'Tačnost iznosi {round(np.mean(results) * 100, 2)}% +/- {round(np.std(results)*100, 2)}%')


# In[43]:


#tacnosti je slicna kao i na prethodnom modelu bez cross-validacije
cross_validation(LogisticRegression(),X,y)


# In[44]:


#ovde je tacnost znacajno bolje u odnosnu na model bez cross-validacije
cross_validation(GaussianNB(),X,y)


# In[45]:


#tacnosti je slicna kao i na prethodnom modelu bez cross-validacije
cross_validation(KNeighborsClassifier(),X,y)


# In[46]:


#rezultati su slicni kao na model koji nije radjen cross-validaticija
cross_validation(DecisionTreeClassifier(),X,y)


# In[47]:


#ovde vidimo da su rezultati bolji kada se poveca dubina stabla ali povecanjem dubine stabla cesto dolazi do teze citljivosti modela
cross_validation(DecisionTreeClassifier(max_depth=5),X,y)


# In[48]:


#funkcija koja pomera granicu odlucivanja (smanjenjem granice se povecava recall a smanjuje precision)
from sklearn.metrics import confusion_matrix
def granica_odlucivanja(algoritam,granica,X_parametar,y_paramatar):
    algoritam.fit(X,y)
    y_hat = algoritam.predict_proba(X)[:, 1] >= granica
    print('===== granica '+str(granica)+"=======")
    print_performance(y,y_hat)
    print(confusion_matrix(y,y_hat))


# In[49]:


#default vrednost za granicu odlucivanja je vec 0.5 pa tako nema nikakvih promena
ocena_trening(LogisticRegression(),X,y)
granica_odlucivanja(LogisticRegression(),0.5,X,y)


# In[50]:


#recall se povecao, smanjio se precision i accuarcy
ocena_trening(LogisticRegression(),X,y)
granica_odlucivanja(LogisticRegression(),0.2,X,y)


# In[51]:


#recall se povecao, smanjio se preciion i accuarcy
ocena_trening(LogisticRegression(),X,y)
granica_odlucivanja(LogisticRegression(),0.1,X,y)


# In[52]:


#recall je ovde dostigao vrednost 1 odnosno sve stvarno positive vrednosti su predvidjene modelom
ocena_trening(KNeighborsClassifier(),X,y)
granica_odlucivanja(KNeighborsClassifier(),0.2,X,y)


# In[53]:


#recall je ovde dostigao vrednost 1 odnosno sve stvarno positive vrednosti su predvidjene modelom ali su accuracy i precision opale
ocena_trening(KNeighborsClassifier(),X,y)
granica_odlucivanja(KNeighborsClassifier(),0.2,X,y)


# In[54]:


#funkcija koja iscrtava ROC AUC krivu i racuna povrsinu ispod nje
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
def roc(algoritam,X_parametar,y_parametar):
    algoritam.fit(X,y)
    print('AUC vrednost je '+str(roc_auc_score(y_parametar,algoritam.predict_proba(X_parametar)[:,1])))
    result = cross_val_score(algoritam,X_parametar,y_parametar,cv=10,scoring='roc_auc')
    print('Cross validation ROC='+str(result.mean())+"+-"+str(result.std()))
    fpr, tpr, thresholds = roc_curve(y, algoritam.predict_proba(X_parametar)[:, 1])
    plt.plot(fpr, tpr)


# In[55]:


#Auc kriva je dobra i vrednost ispod njene povrsine je isto dobra
roc(LogisticRegression(),X,y)


# In[56]:


#ovde nemamo laznih uzbuna jer je model overffitovan. Medjutin za cross-validaciju rezulatt je znacajno losiji.
roc(DecisionTreeClassifier(),X,y)


# In[57]:


roc(GaussianNB(),X,y)


# In[58]:


#Povrsina modela sa cross validacijom i bez nje se razlikuje. Model bez je tacniji jer doslo do overffiting-a.
roc(KNeighborsClassifier(),X,y)


# In[59]:


#kreianje modela i skaliranje podataka MinMax metodom
from sklearn.preprocessing import MinMaxScaler
bank_scaled = X
att_names = list(bank_scaled.columns)


# In[60]:


bank_scaled = MinMaxScaler().fit_transform(bank_scaled)


# In[61]:


from scipy import stats
#prikaz broja odstupanja od standardne devijacije
z = np.abs(stats.zscore(bank_scaled))
z


# In[62]:


#uklanjanje outlajera
bank_scaled = bank_scaled[(z < 3).all(axis=1)]


# In[63]:


#kreiranje KMeans modela sa 2 klastera i maksimalnim brojem iteracija 100
from sklearn.cluster import KMeans
model = KMeans(n_clusters =2, max_iter=100).fit(bank_scaled)
centroids = model.cluster_centers_
#dodavanje klastera instancama
cluster_labels = model.predict(bank_scaled)
#prikaz centroida
centroids


# In[64]:


#prikaz pripadnosti instanci odredjenim klasterima
cluster_labels


# In[65]:


#prikaz Silhouette odnosno SSE, vrednost od priblizno 0.58 je vrlo dobra
from sklearn.metrics import silhouette_score
silhouette_score(bank_scaled,cluster_labels)


# In[66]:


#racunanje udaljenosti unutar klastera
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=100).fit(bank_scaled)
    cluste_labels = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster centers


# In[67]:


#prikaz kretanje unutar klaster razdaljine sa brojem klastera od 1 do 10 odnosno lakat metod
#sa grafikona vidimo da se unutar klastrno rastojanje rapidno smanjuje kada se broj klasterda poveca sa 1 na 2 i sa 2 na 3 a onda porast rastojanja opada i postaje blag
import pandas as pd
from matplotlib import pyplot as plt
plt.close()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Intra Cluster Distance")
plt.show()


# In[68]:


#prikaz kretanja SSE vrednosti modela za broj klastera od 2 do 9
#Sa grafikona vidimo da je najbolja vrednost 2 klastera ali i da ona nije dobra
#Mozemo da pokusamo da izvrsimo algoritam bez dummies vrednosti
from sklearn.metrics import silhouette_score
sse = {}
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, max_iter=100).fit(bank_scaled)
    cluster_labels = kmeans.fit_predict(bank_scaled)
    sse[k] = silhouette_score(bank_scaled, cluster_labels)

plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[69]:


#kreiranje KMeans modela sa 2 klastera i maksimalnim brojem iteracija 100
from sklearn.cluster import KMeans
model = KMeans(n_clusters =2, max_iter=100).fit(bank_scaled)
centroids = model.cluster_centers_
#dodavanje klastera instancama
cluster_labels = model.predict(bank_scaled)
#prikaz centroida
centroids


# In[70]:


bank_scaled_numeric = bank.select_dtypes(include=['int64','float64'])
bank_scaled_numeric = bank_scaled_numeric.drop('y', 1)
bank_scaled_numeric


# In[71]:


bank_scaled_numeric = MinMaxScaler().fit_transform(bank_scaled_numeric)


# In[72]:


#prikaz kretanja SSE vrednosti modela za broj klastera od 2 do 9
#Sa grafikona vidimo da je najbolja vrednost 3 klastera ali i da je 2 zadovoljavajuca
#Vidimo da je ovde sihouette rezultat mnogo bolji nego kad smo koristili i dummy promenjive
from sklearn.metrics import silhouette_score
sse = {}
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, max_iter=100).fit(bank_scaled_numeric)
    cluster_labels = kmeans.fit_predict(bank_scaled_numeric)
    sse[k] = silhouette_score(bank_scaled_numeric, cluster_labels)

plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[73]:


#kreiranje KMeans modela sa 3 klastera i maksimalnim brojem iteracija 100
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 4, max_iter=100).fit(bank_scaled_numeric)
centroids = model.cluster_centers_
#dodavanje klastera instancama
cluster_labels = model.predict(bank_scaled_numeric)
#prikaz centroida
centroids


# In[74]:


#prikaz Silhouette odnosno SSE, vrednost od priblizno 0.58 je vrlo dobra
from sklearn.metrics import silhouette_score
silhouette_score(bank_scaled_numeric,cluster_labels)


# In[75]:


bank['Cluster'] = cluster_labels


# In[76]:


bank.info()


# In[77]:


#kreairanje skupova podataka na osnovu klastera
bank_cluster0 = bank[bank['Cluster'] == 0]
bank_cluster1 = bank[bank['Cluster'] == 1]
bank_cluster2 = bank[bank['Cluster'] == 2]
bank_cluster3 = bank[bank['Cluster'] == 3]


# In[78]:


#pravljenje ulaza i izlaza modela
X0 = bank_cluster0.iloc[:,:-2]
y0 = bank_cluster0.iloc[:,-2]
#kreiranje dummies promenjivih od kategorickih varijabli
pom = X0.select_dtypes(include=['int64','float64'])
for i in categorical:
    X0[i] = X0[i].astype('category')
    dummy = pd.get_dummies(X0[i])
    pom = pd.concat([pom,dummy],axis=1)
X0 = pom


# In[79]:


ocena_test(LogisticRegression(),X0,y0)


# In[80]:


#pravljenje ulaza i izlaza modela
X1 = bank_cluster1.iloc[:,:-2]
y1 = bank_cluster1.iloc[:,-2]
#kreiranje dummies promenjivih od kategorickih varijabli
X1 = X1.select_dtypes(include=['int64','float64'])


# In[81]:


ocena_test(LogisticRegression(),X1,y1)


# In[82]:


#pravljenje ulaza i izlaza modela
X2 = bank_cluster2.iloc[:,:-2]
y2 = bank_cluster2.iloc[:,-2]
#kreiranje dummies promenjivih od kategorickih varijabli
pom = X2.select_dtypes(include=['int64','float64'])
for i in categorical:
    X2[i] = X2[i].astype('category')
    dummy = pd.get_dummies(X2[i])
    pom = pd.concat([pom,dummy],axis=1)
X2 = pom
X.info()


# In[83]:


ocena_test(LogisticRegression(),X2,y2)


# In[84]:


#pravljenje ulaza i izlaza modela
X3 = bank_cluster3.iloc[:,:-2]
y3 = bank_cluster3.iloc[:,-2]
#kreiranje dummies promenjivih od kategorickih varijabli
pom = X3.select_dtypes(include=['int64','float64'])
for i in categorical:
    X3[i] = X3[i].astype('category')
    dummy = pd.get_dummies(X3[i])
    pom = pd.concat([pom,dummy],axis=1)
X3 = pom
X.info()


# In[85]:


ocena_test(LogisticRegression(),X3,y3)


# In[86]:


ocena_test(LogisticRegression(),X,y)


# In[ ]:




