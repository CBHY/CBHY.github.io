# Git 08



## GitHub 연동

### 가입 후 초기설정

#### 	Personal access token 설정

##### 	1.  프로필 -  Settings - Developer Setting

##### 	2. Personal Access tokens - Generate new token

##### 	3. 원하는 기간, 기능 체크 후 Generate token

##### 	4. 토큰 키 복사

##### 	5. 자격 증명 관리자 - Windows  자격 증명 

##### 	6. 일반 자격 증명에 git:https://@github.com 자격 정보 생성

##### 	7. 사용자 이름 : Github 이름

##### 	8. 암호 : 토큰 키



### GitHub Repository 생성 후 연결

```bash
git remote add origin (주소(Https))
```

#### 	Git의 원격 저장소를 추가, 이름은 Origin

```bash
git push -u origin main
# 또는
git push --set-upstream origin main
```

#### 	로컬 저장소의 커밋 내역을 원격으로 push

#### 	main과 origin을 연결(upstream)

```bash
# 원격 Branch 목록 확인
git remote
```

```bash
# 원격 Branch 삭제
git remote remove (branch)
```



### Gitbub에서 프로젝트 다운로드

```bash
git clone (주소)
```

