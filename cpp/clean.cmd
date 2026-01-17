@echo off
echo ===========================================
echo C++ Build Artifacts Cleaning...
echo ===========================================

:: 삭제할 폴더 목록 (현재 폴더 및 하위 폴더 전체 순회)
for /d /r . %%d in (.vs Debug Release x64 x86 ipch) do (
    if exist "%%d" (
        echo [Deleting Folder] "%%d"
        rd /s /q "%%d"
    )
)

:: 삭제할 파일 확장자 목록
del /s /q *.pdb *.idb *.ilk *.sdf *.ipch *.log *.obj *.tlog *.lastbuildstate *.unsuccessfulbuild

echo ===========================================
echo Cleaning Complete!
echo ===========================================
pause