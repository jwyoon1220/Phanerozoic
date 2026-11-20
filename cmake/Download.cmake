# cmake -DURL=... -DDEST=... -P Download.cmake
file(DOWNLOAD ${URL} ${DEST} STATUS status SHOW_PROGRESS)
list(GET status 0 code)
if (NOT code EQUAL 0)
    file(REMOVE ${DEST})
    message(FATAL_ERROR "Download of ${URL} failed: ${status}")
endif ()
