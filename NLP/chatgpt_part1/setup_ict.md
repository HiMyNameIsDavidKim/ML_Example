파이썬 3.9.8 이상의 버전을 사용  
저는 3.9.13 버전을 사용하였으며 [여기](https://www.python.org/downloads/release/python-3917/)에서 신규 버전 다운로드 및 설치 후 1번부터 진행해주시면 됩니다.



### 1. Poetry 설치
가상환경과 동일하게 구성하기 위해 `poetry`를 활용합니다.

아래와 같은 방법으로 설치하면 됩니다.
- 리눅스/맥/윈도우(WSL): `curl -sSL https://install.python-poetry.org | python3 -`
- 윈도우(Powershell): `(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -`

잘 설치되었을 경우 `poetry --version` 를 실행하였을 때 `Poetry (version 1.*.*)`와 같은 문구가 출력됩니다.

만약 Poetry가 익숙하지 않고 설치에 어려움을 겪으실 경우를 대비하여 requirements.txt 도 함께 동봉

`2. Poetry 가상 환경 설치 및 활성화` 는 건너뛰고 `pip install -r requirements.txt` 를 통해 익숙한 방법으로 설치해주셔도 괜찮습니다.

---

### 2. Poetry 가상 환경 설치 및 활성화

지금 이 md 파일이 있는 위치로 이동 후 `poetry install` 을 하시게 되면 저와 동일한 파이썬 패키지들로 이루어진 가상환경이 생성됩니다.

이후 `poetry shell` 을 입력하면 가상환경이 활성화 됩니다.
`deactivate` 를 입력하면 가상환경이 비활성화 됩니다.

터미널 혹은 Powershell은 새로운 윈도우를 열 때마다 가상환경이 초기화 되니 `poetry shell` 로 계속 활성화 해주셔야 합니다.

---

### 3. Streamlit 실행
강의에서 챗봇 화면 등의 실습 UI를 위해 Streamlit 패키지를 사용하고 있습니다.  
`poetry` 를 통해서 잘 설치되지 않는 사례가 많기 때문에,  
`poetry shell`로 가상환경을 활성화 하고, `pip install streamlit==1.25.0` 를 통해 설치해주시면 됩니다.

강의에서 사용하는 ui 파일의 목록은 아래와 같습니다.
- `part02/_ui/app.py`: 기본적인 챗봇 ui 입니다.
- `part05/_ui/gen1_app.py`: 소설 작성 1세대 실습을 위한 ui 입니다.
- `part05/_ui/gen2_app.py`: 소설 작성 2세대 실습을 위한 ui 입니다.


각 파일의 실행 방법은 `python -m streamlit run {파일 경로}` 입니다.

---

### 4. FastAPI 실행
강의에서 챗봇 프론트엔드와 통신하기 위해 백엔드로 FastAPI를 사용하고 있습니다.

강의에서 사용하는 FastAPI 백엔드를 실행하는 파일 목록은 아래와 같습니다.
- `part02/ch03/generation_1.py`
- `part02/ch03/generation_2.py`
- `part02/ch03/generation_3.py`
- `part05/ch03/gen1/api.py`
- `part05/ch03_no_framework/gen2/api.py`
- `part05/ch04_langchain/gen1/api.py`
- `part05/ch04_langchain/gen2/api.py`
- `part05/ch05_semantic_kernel/gen1/api.py`
- `part05/ch05_semantic_kernel/gen2/api.py`
  
각 파일의 실행 방법은 `python {파일 경로}` 로 실행해도 되고, 강의에서와 같이 `uvicorn {파일 경로}:app --reload`로 해주셔도 괜찮습니다.  
주의점은 파일 경로를 "." 로 이어주어야 합니다.  
예를 들어
- `part02` 디렉터리에 있다면 `uvicorn chapter03.generation_3:app --reload` 로 실행
- 하위 디렉터리에 있다면 `uvicorn part02.chapter03.generation_3:app --reload` 로 실행
  


