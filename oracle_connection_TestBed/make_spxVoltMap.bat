set python64=C:/Program Files/Python39
set PATH=%python64%/Scripts/; %python64%/;%PATH%
cd ../bin/spx_viewer
del spxVoltMap.exe
cd ../../src/EMS_HOST/PSAPP/TSE/spxVoltMap
rmdir /S /Q dist
rmdir /S /Q build
del *.spec
set python_lib_path=C:/Program Files/Python39/Lib/site-packages
pyinstaller --log-level=INFO ^
	--windowed --onefile ^
	--noupx --clean^
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
	--hidden-import=scipy.interploate ^
	--hidden-import=pykrige.ok ^
	--hidden-import=pykrige.core ^
	--hidden-import=pykrige.variogram_models ^
	--hidden-import=pykrige.kriging_tools ^
	--hidden-import=matplotlib ^
	--hidden-import=typing ^
	--hidden-import=collections.abs ^
	--collect-all pykrige ^
	--collect-all scipy ^
	--add-data "spxVoltMap.ui;." ^
	--add-data "shp.geojson;." ^
	--add-data "jeju_coastline.csv;." ^
	--add-data "land_coastline.csv;." ^
	--add-data "%python_lib_path%/numpy;./" ^
	--add-data "%python_lib_path%/numpy.libs;./" ^
	--add-data "%python_lib_path%/numpy/f2py;./" ^
	--add-data "%python_lib_path%/scipy;./" ^
	--add-data "%python_lib_path%/scipy/interpolate;./" ^
	--debug=all ^
spxVoltMap.py > build_log.txt 2>&1
cd dist
move spxVoltMap.exe ../../../../../../bin/spx_viewer
cd ..
rmdir /S /Q dist
rmdir /S /Q build
del *.spec