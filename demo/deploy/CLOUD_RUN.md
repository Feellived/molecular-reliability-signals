## Cloud Run 배포 안내

현재 배포 주소는 https://mist-700016601256.asia-northeast3.run.app 이며 프로젝트는 mist-demo-507509다.

Google Cloud Run은 Dockerfile을 그대로 받아 원격에서 빌드하고 실행한다. 로컬에 Docker를 설치할 필요가 없다.

무료 한도는 월 요청 200만 건, vCPU 18만 초, 메모리 36만 GB초다. 최소 인스턴스를 0으로 두면 접속이 없을 때 완전히 잠들어 과금이 발생하지 않는다. 데모 수준 트래픽에서는 한도 안에 들어온다. 다만 계정 생성 시 카드 등록은 필요하다.

### 1. 계정과 프로젝트

1. console.cloud.google.com 접속 후 Google 계정으로 로그인
2. 결제 계정 등록. 신규 가입이면 90일 300달러 크레딧이 붙는다
3. 상단 프로젝트 선택기에서 새 프로젝트 생성. 이름은 `mist-demo` 정도

### 2. gcloud 설치

```
brew install --cask google-cloud-sdk
```

설치 후 로그인하고 프로젝트를 지정한다. `PROJECT_ID`는 콘솔 대시보드에서 확인한다.

```
gcloud auth login
gcloud config set project PROJECT_ID
```

### 3. 필요한 API 켜기

```
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### 4. 배포

배포 묶음을 만든 뒤 그 폴더에서 실행한다.

```
python deploy/build_space.py \
  --bundle ../data/processed/scores_role4/demo_bundle \
  --chemberta ../../Jiye/checkpoints/chemberta \
  --out /tmp/mist-space

cd /tmp/mist-space
gcloud run deploy mist \
  --source . \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --min-instances 0
```

첫 실행은 빌드에 5분에서 10분 걸린다. torch를 받는 시간이다. 끝나면 `https://mist-...run.app` 형태의 주소가 나온다.

옵션의 뜻은 이렇다. `--region asia-northeast3`은 서울이라 지연이 짧다. `--memory 2Gi`는 지문 모델과 언어 모델을 함께 올리기 위한 값으로, 1Gi로 줄이면 물성을 여러 개 부를 때 죽을 수 있다. `--timeout 300`은 첫 요청에서 모델을 읽는 시간을 감당하기 위한 값이다. `--min-instances 0`이 과금을 막는 핵심이며, 대신 한동안 접속이 없으면 다음 첫 요청이 느려진다.

### 5. 다시 올리기

같은 명령을 다시 실행하면 새 판이 배포된다.

```
cd /tmp/mist-space && gcloud run deploy mist --source . --region asia-northeast3
```

### 비용을 확실히 막으려면

콘솔의 결제 항목에서 예산 알림을 걸어둔다. 1달러만 넘어도 메일이 오게 해두면 예상 밖의 과금을 바로 알 수 있다.

발표가 끝나면 서비스를 지운다.

```
gcloud run services delete mist --region asia-northeast3
```

### 참고

포트는 Cloud Run이 `PORT` 환경변수로 넘기며 Dockerfile이 그 값을 따른다. 환경변수가 없으면 7860으로 떨어지므로 다른 호스트에서도 같은 이미지가 돈다.

빌드가 실패하면 로그를 이렇게 본다. 컨테이너가 뜨지 못한 경우는 빌드 로그가 아니라 리비전 로그를 봐야 하며, 역순으로 나오므로 `--order=asc`를 붙여야 실제 예외가 보인다.

```
gcloud builds log $(gcloud builds list --region asia-northeast3 --limit 1 --format="value(id)") \
  --region asia-northeast3

gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="mist"' \
  --limit 100 --order=asc --format="value(textPayload)" | tail -20
```

의존성을 올릴 때는 두 가지를 먼저 확인한다. 파이썬 판이 PyPI에 리눅스 x86_64 휠로 실재하는지, 그리고 새로 추가한 패키지가 요구하는 시스템 라이브러리가 Dockerfile의 apt 목록에 있는지다. 둘 다 conda 환경에서는 드러나지 않는다.
