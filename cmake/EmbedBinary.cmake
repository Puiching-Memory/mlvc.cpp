if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT DEFINED SYMBOL)
    message(FATAL_ERROR "EmbedBinary.cmake requires INPUT, OUTPUT, and SYMBOL")
endif()

file(READ "${INPUT}" binary_hex HEX)
string(REGEX REPLACE "([0-9a-fA-F][0-9a-fA-F])" "0x\\1," binary_data
       "${binary_hex}")

file(WRITE "${OUTPUT}"
"#include <cstddef>\n"
"extern \"C\" alignas(16) const unsigned char ${SYMBOL}[] = {${binary_data}};\n"
"extern \"C\" const std::size_t ${SYMBOL}_size = sizeof(${SYMBOL});\n")
