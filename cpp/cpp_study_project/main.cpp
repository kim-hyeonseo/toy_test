#include "pch.h"

// namespace 를 사용함으로써 외부에서 해당 함수는 사용하지 않는구나를 알 수 있음
// 다른 대안으로 앞에 static 을 붙여주는 방법이 있음


namespace {
// IO 시험
 void IO_function() {

    std::cout << "Enter two numbers: ";

    int x{};
    std::cin >> x;

    int y{};
    std::cin >> y;

    std::cout << "You entered " << x << " and " << y << '\n';

}

// int 일 때와 unsigned int 일때 결과값이 다름
// int 최댓값이 21억 xxxx 이기 때문에.. 
 void foo(unsigned int x) 
{
    std::cout << x << '\n';
}


 long getCPPStandard() {
    // Visual Studio is non-conforming in support for __cplusplus (unless you set a specific compiler flag, which you probably haven't)
    // In Visual Studio 2015 or newer we can use _MSVC_LANG instead
    // See https://devblogs.microsoft.com/cppblog/msvc-now-correctly-reports-__cplusplus/
#if defined (_MSVC_LANG)
    return _MSVC_LANG;

#elif defined (_MSC_VER)
    // If we're using an older version of Visual Studio, bail out
    return -1;
#else
    // __cplusplus is the intended way to query the language standard code (as defined by the language standards)
    return __cplusplus;
#endif
}
 void compiler_check() {
    // This program prints the C++ language standard your compiler is currently using
    // Freely redistributable, courtesy of learncpp.com (https://www.learncpp.com/cpp-tutorial/what-language-standard-is-my-compiler-using/)



    const int numStandards = 7;
    // The C++26 stdCode is a placeholder since the exact code won't be determined until the standard is finalized
    const long stdCode[numStandards] = { 199711L, 201103L, 201402L, 201703L, 202002L, 202302L, 202612L };
    const char* stdName[numStandards] = { "Pre-C++11", "C++11", "C++14", "C++17", "C++20", "C++23", "C++26" };



    long standard = getCPPStandard();

    if (standard == -1)
    {
        std::cout << "Error: Unable to determine your language standard.  Sorry.\n";
    }

    for (int i = 0; i < numStandards; ++i)
    {
        // If the reported version is one of the finalized standard codes
        // then we know exactly what version the compiler is running
        if (standard == stdCode[i])
        {
            std::cout << "Your compiler is using " << stdName[i]
                << " (language standard code " << standard << "L)\n";
            break;
        }

        // If the reported version is between two finalized standard codes,
        // this must be a preview / experimental support for the next upcoming version.
        if (standard < stdCode[i])
        {
            std::cout << "Your compiler is using a preview/pre-release of " << stdName[i]
                << " (language standard code " << standard << "L)\n";
            break;
        }
    }



}
 
 
 void walk_the_vect() {
    std::vector<int> vect;
    for (int count = 0; count < 6; ++count)
        vect.push_back(count);


    //container::iterator는 읽기/쓰기 이터레이터를 제공합니다.
    std::vector<int>::const_iterator it; // declare a read-only iterator
    it = vect.cbegin(); // assign it to the start of the vector
    while (it != vect.cend()) // while it hasn't reach the end
    {
        std::cout << *it << ' '; // print the value of the element it points to
        ++it; // and iterate to the next element
    }

    std::cout << '\n';
}

 void walk_the_list() {
    std::list<int> li;
    for (int count = 0; count < 6; ++count)
        li.push_back(count);

    std::list<int>::const_iterator it; // declare an iterator
    it = li.cbegin(); // assign it to the start of the list
    while (it != li.cend()) // while it hasn't reach the end
    {
        std::cout << *it << ' '; // print the value of the element it points to
        ++it; // and iterate to the next element
    }

    std::cout << '\n\n';
}

 void walk_the_set() {
        std::set<int> myset;
        myset.insert(7);
        myset.insert(2);
        myset.insert(-6);
        myset.insert(8);
        myset.insert(1);
        myset.insert(-4);

        std::set<int>::const_iterator it; // declare an iterator
        it = myset.cbegin(); // assign it to the start of the set
        while (it != myset.cend()) // while it hasn't reach the end
        {
            std::cout << *it << ' '; // print the value of the element it points to
            ++it; // and iterate to the next element
        }

        std::cout << '\n';
        }

  }


int main() {
//    IO_function();
//    unsigned int x{ 2200000000 }; 
//    foo(x);  
//    compiler_check();
//    walk_the_vect();
//    walk_the_list();
    walk_the_set();
	return 0;
}   