# Git 03



## Git의 세가지 공간 이해

![KakaoTalk_20220219_155830369](C:\Users\kkhhy\Desktop\KakaoTalk_20220219_155830369.jpg)

### Working directory

#### 	local directory에서 작업하는 것들이 담겨있는 현제 directory

##### 		untracked : git에서 관리하지 않는 파일(.gitignore), git에서 관리한 적 없는 파일(새로운 파일)

##### 		tracked : git에서 관리하고 있거나, 이전에 관리한 적이 있는 파일, 변화가 생기면 표시



### Staging area

#### 	commit 이전에 준비 단계, working dirctory에서 git add로 넣고, git restore --staged로 뺄 수 있다.	

```bash
# file.c를 working directory에서 stagig area로 이동
git add file.c
```

```bash
# working directory의 모든 파일을 stagig area로 이동(.gitignore 제외)
git add .
```

```bash
# file.c를 stagig area에서 working directory로 이동
git restore --staged file.c
```



### Repository

#### 	commit단위로 저장되는 최종 단계, .git directory라고도 함. repository상태에 저장된 파일을 local에서 수정하거나 파일을 추가하면, working driectory에 표시한다

```bash
# 이전 repository상태로 돌아가기 > git 05에서 설명(Reset, revert)
```

```bash
# Repository에서 staging area로 돌아가기(Commit 취소)
git reset --soft
```

```bash
# Repository에서 Working driectory로 돌아가기(Add, Commit 둘 다 취소)
git reset (--mixed)
```

#### 	reset의 default는  --mixed, 이 코드를 써서 돌아갔다고 해도 Working driectory에서의 변화는 유지



## 공간 이해 최종 정리

![제목 없는 노트북 1 P1](C:\Users\kkhhy\Desktop\제목 없는 노트북 1 P1.png)
