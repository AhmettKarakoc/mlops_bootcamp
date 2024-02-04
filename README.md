# Tech Stack
- Python
- Poetry for Dependency Management
- Typer for CLI Application
- BigQuery as a Data Lake
- Docker for Deployment
- GKE as Production Environment
- GitLab for CI-CD
- GitLab for Code Repository

# Creating a New Project
- Make sure poetry is installed in the development environment.
    - <code> poetry --version </code>
    - If not it can be installed via two of the below methods:
        1. <code> curl -sSL https://install.python-poetry.org | python3 - </code>
        2. <code> pip install poetry </code>
- Make sure virtual environment is created under project directory.
    - <code> poetry config --list </code> -> virtualenvs.in-project = true
    - If above config is set to false; then execute below command.
    - <code> poetry config virtualenvs.in-project true </code>
- <code> poetry init </code>
    - This command triggers an interactive dependency initiation sequence. You can answer all question appropriately.
- <code> poetry add pandas </code> : To add a dependency to both development and production environment.
- <code> poetry add --dev pandas </code> : To add a dependency to only development environment. These librarires will be discarded into production deployment phase (Docker Image Building Step)


# Steps for Creating New Project

1) Install cookiecutter.
   1) pip3 install cookiecutter
   2) brew install cookiecutter
2) cookiecutter <your_project_url>
3) Make arrangments in pyproject.toml
4) poetry install
5) source .venv/bin/activate || poetry shell
6) pre-commit install

Data Science Project Template
==============================


Project Organization
------------

    ├── data               <- Proje içerisinde saklanmak istenen csv, json, excel vb. gibi dataların tutulduğu klasördür.  (5 MB üzerindeki datalar GIT' e atılmayacak.)
    │   │		      Aynı şekide projenin çıktısı (son çıktı veya ara çıktı) bir data saklanmak isteniyorsa yeri bu klasör olacaktır. Alt klasörlerin içeriği şu şekildedir.
    │   │
    │   │   
    │	├── external       <- Örnek : Projede kullanılmak üzere oluşturulmuuş bir look-up tablosunun csv hali
    │	├── processed      <- Örnek : Projenin ara adımlarında oluşturulan dataların excel halinde saklanan hali  
    │	├── output         <- Örnek : Projenin çıktılarını içeren excel dosyaları  
    │	├── raw		   <- Örnek : Hive, Mongo vb gibi bir db' den çekilmiş bir raw tablonun csv hali.	      	    │
    │
    ├── docs               <- Projeye referans gösterilebilecek makale, internet sayfası, kütüphane gibi bilgilerin tutulduğu klasördür.
    │
    │
    ├── models             <- Projenin ana veya ara çıktıları olarak üretilen model, optimizer, embedding gibi objelerin
    │			      pickle, h5 vb. gibi formatlarda tekrar kullanılabilir şekilde saklandığı klasördür. (5 MB altındaki modeller GIT' e atılacak. Daha büyük modeller bucket' a yaratılacak)
    │			      Model dosyalarının GCP bucket' a otomatik atılması CI/CD içerisinde ele alınacak (otomasyonla birlikte tüm modeller için bucket' a gidilecek)
    │
    ├── notebooks          <- Projenin geliştirilmesinde, kullanılan notebook' ların free format tutulduğu bir klasödür. 
    │			      Bu klasörü projenin müsvedde bölümü olarak düşünebiliriz, çünkü notebook' larda geliştirilen kodlar her zaman src klasörü altındaki 
    │			      .py formatlarına çekilerek teste ve production' a iletilecektir. (Bu kodların git' e atılması ds' in kendi kararıdır.Özellikle nice-to-have büyük notebook' ların atılmaması tercihimizdir.) 
    │
    │
    ├── reports            <- Projede oluşturulan excel, word, pdf gibi raporlama araçlarının veya kod çalışırken üretilen logların tutulduğu klasördür.
    │	├── report_docs    <- Raporlama amaçlı oluşturulan dosyalar ve figure' lerin tutulduğu klasördür
    │	├── logs           <- Log dosyalarının tutulduğu klasördür. 
    │
    │
    │
    ├── src                <- Projeyi oluşturan source code' lar bu klasörün altındaki ilgili alt klasörlerde yer alır.
    │   │   
    │	├── config         <- Burada projede parametrik olarak tanımlanması gereken db adı, kafka topic' i, API adresi vb. dev, test, prod ortmalarında farklılaşabilecek
    │   │		      değerler config.yml dosyasına manuel olarak tanımlanır. Geliştirme kodları içerisinde get_env_configs() fonksiyonu çağrılarak makine IP'isne göre
    │   │		      içinde bulunan ortamın configleri bir dictionary olarak alınır ve kullanılır.
    │   │
    │   │    
    │   │
    │   ├── data           <- Projedeki veri kaynakları ile veri alış verişi için kullanılan connection script' lerinin 
    │   │   │                 ve .sql, .hql vb gibi data handling scriptlerinin tutulduğu klasördür
    │   │   │
    │   │   └── bigquery      <- Projelerin BigQuery üzerinde koşulacak querylerin bulunduğu dosyalar bu dizin altında yer alır.
    │   │
    │   │
    │   │
    │   ├── features       <- Modellerde kullanılacak feature' ların oluşturulma adımlarında kullanılacak .py dosyalarını içeren klasördür.
    │   │   
    │   │
    │   ├── models         <- Modellerinin eğitim ve skorlama fonksiyonlarının tanımlandığı .py  kodlarının bulunduğu klasördür.
    │   │   
    │   │
    │   ├── product_requirements  <-  Diğer klasörlerin kapsamına girmeyen tüm geliştme kodları bu klasörde saklanır. 
    │   │ 				Örnek: Exception handling,  post process işlemler, API call' lar , skor hesaplama adımları, ürün filtreleme işlemleri vb.
    │   │          
    │   │
    │   ├── utils                 <- Projede kullanılan yardımcı fonskiyonları tutulduğu klasördür.(Örnek: son versiyonu alma, modeli ilgili path' e save etme vb.)
    │   │
    │   │ 
    │   ├── visualization  <- Proje ile ilgili görselleştirme kodlarının tutulduğu bölümdür.
    │   │       
    │   │
    │   ├── __init__.py    <- src klasörünü bir Python modulü haline getirir. Bu sayede "import from src...." şeklinde kullanılabilir.  
    │
    ├── tests              <- Proje için yapılan testlerde kullanılan notebook' ların tutuldu bölümdür. Free-format tutulur. Bu bölümde
    │			      Bu bölüm de test ve review tasklarının müsveddesi olarak düşünülebilir. Buradaki kodlar git' e doğal akışta gitmeyeceği için
    │			      pair ds bu kodların tekrar kullanılabilir olmasını istiyorsa, git reposunda geçici bir branch açarak test kodlarını bu branch' te yedekleyebilir.
    │
    ├── README.md          <- Projedeki klasör yapısının tanıtıldığı ve hangi klasörde hangi tipte dokümanların bulunması gerektiğinin tariflendiği dokümandır
    │
    ├── ds_app.py          <- Projeyi çalıştıran ana CLI app yapısındaki scripttir. 



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
