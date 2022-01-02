#ifndef UTILS_HPP_
#define UTILS_HPP_

namespace diff {
    namespace utils {

        typedef struct matsz {

            int height;
            int width;

            matsz(int h, int w) : height(h), width(w) {}
            matsz(): matsz(0, 0) {}
            int area();

        } matsz;

        template <class T> 
        static inline void swap(T& a, T& b) {
            T tmp;

            tmp = a;
            a = b;
            b = tmp;
        }

    };
}

#endif