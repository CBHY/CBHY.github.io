# Git 07



## Branch 합치기

###  Merge

#### 	두 branch를 한 commit에 이어 붙이는 방법, branch의 사용 내역이 남기고 싶은 경우 적합

```bash
# main Branch로 merge
git switch main # main branch로 이동
git merge (branch)
```

#### 	merge도 하나의 Commit으로 저장되기 때문에, git reset으로 되돌리기가 가능함.

```bash
# merge된 Branch 삭제
git branch -d branch
```



### Rebase

#### 	branch를 다른 branch에 이어 붙이는 방법, branch의 구성이 간결함

##### 	** Merge와는 반대로, 합쳐야 할 Branch에서 진행 **

```bash
# main branch로 rebase
git switch (branch) # main branch로 rebase 할 branch로 이동
git rebase main
```

#### 	rebase를 진행하면, main에 있는 commit모두 진행 이후, 합쳐진 branch의 commit이 진행 되므로, main을 최신화 시켜주어야 main으로 합쳐지는 의미임

```bash
# main branch로 merge
git switch main
git merge (branch)
```





## Branch 충돌

#### branch merge, rebase하는 과정에서, branch내에서 파일 등의 변화가 충돌하면, 충돌 해결한 이후 merge, rebase를 continue하는 과정이 필요함.



### Merge 충돌

```bash
# main branch로 merge
git switch main
git merge (branch)
```

#### 	충돌 발생 -  오류 메세지로 확인

```bash
git status
```

##### 	git status로도 확인 가능

```bash
# 당장 충돌 해결이 힘들다면 merge 중단
git merge --abort
```

```bash
# 충돌 부분 수정 후
git add .
git commit
```



### Rebase 충돌

```bash
# main branch로 rebase
git switch (branch)
git rebase main
```

#### 	충돌 발생 -  오류 메세지로 확인

```bash
git status
```

##### 	git status로도 확인 가능

```bash
# 당장 충돌 해결이 힘들다면 rebase 중단
git rebase --abort
```

```bash
# 충돌 부분 수정 후
git rebase --continue
```

