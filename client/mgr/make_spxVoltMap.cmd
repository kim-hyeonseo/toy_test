@echo off
setlocal

set python64=C:\Program Files\Python39
set PATH=%python64%\Scripts\; %python64%\;%PATH%
cd ..\bin\HistViewer
rmdir /S /Q spxVoltMap
cd ..\..\src\EMS_HOST\PSAPP\TSE\spxVoltMap
rmdir /S /Q dist
rmdir /S /Q build
del *.spec
set python_lib_path=C:\Program Files\Python39\Lib\site-packages
pyinstaller --windowed --onedir^
	--icon="..\..\..\..\..\bin\HistViewer\Viewport.ico"^
	--collect-all spxVoltMap.py^
	--noupx --clean ^
	--hidden-import=fileinput ^
	--hidden-import=numpy ^
	--hidden-import=numpy.f2py ^
	--hidden-import=numpy.f2py.f2py2e ^
	--hidden-import=numpy.f2py.crackfortran ^
	--hidden-import=numpy.f2py.__version__ ^
	--hidden-import=numpy.f2py.auxfuncs ^
	--hidden-import=numpy.f2py.cfuncs ^
	--hidden-import=numpy.f2py.symbolic ^
	--hidden-import=numpy.f2py.rules ^
	--hidden-import=numpy.f2py.capi_maps ^
	--hidden-import=numpy.f2py.cb_rules ^
	--hidden-import=numpy.f2py._isocbind ^
	--hidden-import=numpy.f2py.common_rules ^
	--hidden-import=numpy.f2py.func2subr ^
	--hidden-import=numpy.f2py.use_rules ^
	--hidden-import=numpy.f2py.f90mod_rules ^
	--hidden-import=numpy.f2py._backends ^
	--hidden-import=numpy.f2py.diagnose ^
	--hidden-import=scipy ^
	--hidden-import=scipy.spatial.ckdtree ^
	--hidden-import=scipy.spatial.qhull ^
	--hidden-import=scipy.spatial.transform ^
	--hidden-import=scipy.special ^
	--hidden-import=scipy.interpolate ^
	--hidden-import=pykrige.ok ^
	--hidden-import=pykrige.core ^
	--hidden-import=pykrige.variogram_models ^
	--hidden-import=pykrige.kriging_tools ^
	--hidden-import=matplotlib ^
	--hidden-import=typing ^
	--hidden-import=collections.abc ^
	--collect-all pykrige ^
	--collect-all scipy ^
	--add-data "load_line.gif;." ^
    --add-data "land.geojson;." ^
    --add-data "optimized.geojson;." ^
    --add-data "jeju.geojson;." ^
    --add-data "shp.geojson;." ^
    --add-data "voltmap_local.csv;." ^
	--add-data "assets;assets" ^
    --add-data "%python_lib_path%\zstandard;zstandard" ^
    --add-data "%python_lib_path%\zstandard-0.21.0.dist-info;zstandard-0.21.0.dist-info" ^
    --add-data "%python_lib_path%\numpy;numpy" ^
    --add-data "%python_lib_path%\numpy\.libs;numpy/.libs" ^
    --add-data "%python_lib_path%\scipy;scipy" ^
    --paths="%python_lib_path%\typing" ^
spxVoltMap.py

:: 3. 빌드 성공 여부 체크
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyInstaller 빌드에 실패했습니다.
    pause
    exit /b %ERRORLEVEL%
)
:: 3. 빌드 성공 여부 체크
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyInstaller 빌드에 실패했습니다.
    pause
    exit /b %ERRORLEVEL%
)

cd dist\spxVoltMap
if not exist _internal mkdir _internal

for /f "delims=" %%i in ('dir /b /a ^| findstr /v /i "spxVoltMap.exe _internal"') do (
    move /Y "%%i" _internal\
)

cd ..
move /Y spxVoltMap ..\..\..\..\..\..\bin\HistViewer 
cd ..\..\..\..\..\..\mgr

echo [SUCCESS] 빌드 및 정리가 완료되었습니다.
endlocal