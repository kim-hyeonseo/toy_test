# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('jeju.geojson', '.'), ('shp.geojson', '.'), ('optimized.geojson', '.'), ('assets', 'assets'), ('C:\\Program Files\\Python39\\Lib\\site-packages\\numpy', '.\\numpy'), ('C:\\Program Files\\Python39\\Lib\\site-packages\\numpy\\.libs', '.\\numpy.libs'), ('C:\\Program Files\\Python39\\Lib\\site-packages\\numpy\\f2py', '.\\numpy\\f2py'), ('C:\\Program Files\\Python39\\Lib\\site-packages\\zstandard', '.\\zstandard'), ('C:\\Program Files\\Python39\\Lib\\site-packages\\zstandard-0.21.0.dist-info', '.\\zstandard-0.21.0.dist-info'), ('C:\\Program Files\\Python39\\Lib\\site-packages\\scipy', '.\\scipy'), ('C:\\Program Files\\Python39\\Lib\\site-packages\\scipy\\interpolate', '.\\scipy\\interpolate')]
binaries = []
hiddenimports = ['fileinput', 'numpy', 'numpy.f2py', 'numpy.f2py.f2py2e', 'numpy.f2py.crackfortran', 'numpy.f2py.__version__', 'numpy.f2py.auxfuncs', 'numpy.f2py.cfuncs', 'numpy.f2py.symbolic', 'numpy.f2py.rules', 'numpy.f2py.capi_maps', 'numpy.f2py.cb_rules', 'numpy.f2py._isocbind', 'numpy.f2py.common_rules', 'numpy.f2py.func2subr', 'numpy.f2py.use_rules', 'numpy.f2py.f90mod_rules', 'numpy.f2py._backends', 'numpy.f2py.diagnose', 'scipy', 'scipy.spatial.ckdtree', 'scipy.spatial.qhull', 'scipy.spatial.transform', 'scipy.special', 'scipy.interpolate', 'pykrige.ok', 'pykrige.core', 'pykrige.variogram_models', 'pykrige.kriging_tools', 'matplotlib', 'typing', 'collections.abc']
tmp_ret = collect_all('spxVoltMap_dash_app.py')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pykrige')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


block_cipher = None


a = Analysis(
    ['spxVoltMap_dash_app.py'],
    pathex=['C:\\Program Files\\Python39\\Lib\\site-packages\\typing'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='spxVoltMap_dash_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='spxVoltMap_dash_app',
)
