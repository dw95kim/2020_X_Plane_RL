# Install script for directory: C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/XPlaneConnect")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect/64" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/Debug/lin.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect/64" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/Release/lin.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect/64" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/MinSizeRel/lin.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect/64" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/RelWithDebInfo/lin.lib")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect/64" TYPE SHARED_LIBRARY FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/Debug/lin.xpl")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect/64" TYPE SHARED_LIBRARY FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/Release/lin.xpl")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect/64" TYPE SHARED_LIBRARY FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/MinSizeRel/lin.xpl")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect/64" TYPE SHARED_LIBRARY FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/RelWithDebInfo/lin.xpl")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/Debug/lin.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/Release/lin.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/MinSizeRel/lin.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/RelWithDebInfo/lin.lib")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect" TYPE SHARED_LIBRARY FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/Debug/lin.xpl")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect" TYPE SHARED_LIBRARY FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/Release/lin.xpl")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect" TYPE SHARED_LIBRARY FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/MinSizeRel/lin.xpl")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/XPlaneConnect" TYPE SHARED_LIBRARY FILES "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/RelWithDebInfo/lin.xpl")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "C:/Users/DWKim/Desktop/X_plane/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/xpcPlugin/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
