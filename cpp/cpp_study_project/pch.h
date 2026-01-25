// pch.h: 미리 컴파일된 헤더 파일입니다.
// 아래 나열된 파일은 한 번만 컴파일되었으며, 향후 빌드에 대한 빌드 성능을 향상합니다.
// 코드 컴파일 및 여러 코드 검색 기능을 포함하여 IntelliSense 성능에도 영향을 미칩니다.
// 그러나 여기에 나열된 파일은 빌드 간 업데이트되는 경우 모두 다시 컴파일됩니다.
// 여기에 자주 업데이트할 파일을 추가하지 마세요. 그러면 성능이 저하됩니다.


// 프로젝트 속성 / 'c/c++' / 미리컴파일된 헤더 / 미리컴파일된 헤더 : 사용(/Yu)로 수정
// 미리 컴파일된 헤더 파일이름을 pch.h 파일과 동일하게 설정
// x64 하위 폴더 중 Debug 아래 cpp_study_project.pch 파일을 생성

#pragma once  // 중복 포함 방지 필수!

#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // std::min_element and std::max_element
#include <map>
#include <list>
#include <set>
#include <numeric> // std::iota
#include <format>


/*
과거에는 이렇게 include guard 를 사용

#ifndef PCH_H
#define PCH_H

// 여기에 미리 컴파일하려는 헤더 추가

#endif //PCH_H
*/