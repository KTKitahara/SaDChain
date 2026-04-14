# Microsoft Developer Studio Project File - Name="CCGPHH" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=CCGPHH - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "CCGPHH.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "CCGPHH.mak" CFG="CCGPHH - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "CCGPHH - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "CCGPHH - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 1
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "CCGPHH - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /FR /YX /FD /c
# ADD BASE RSC /l 0x804 /d "NDEBUG"
# ADD RSC /l 0x804 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386

!ELSEIF  "$(CFG)" == "CCGPHH - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD BASE RSC /l 0x804 /d "_DEBUG"
# ADD RSC /l 0x804 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept

!ENDIF 

# Begin Target

# Name "CCGPHH - Win32 Release"
# Name "CCGPHH - Win32 Debug"
# Begin Source File

SOURCE=.\CCGPHH.cpp
# End Source File
# Begin Source File

SOURCE=.\CCGPHHFitness.cpp
# End Source File
# Begin Source File

SOURCE=.\cmp.cpp
# End Source File
# Begin Source File

SOURCE=.\comfunc.h
# End Source File
# Begin Source File

SOURCE=.\crossover.cpp
# End Source File
# Begin Source File

SOURCE=.\decode.cpp
# End Source File
# Begin Source File

SOURCE=.\GPHH.CPP
# End Source File
# Begin Source File

SOURCE=.\LowLevel_heuristics.cpp
# End Source File
# Begin Source File

SOURCE=.\max.cpp
# End Source File
# Begin Source File

SOURCE=.\min.cpp
# End Source File
# Begin Source File

SOURCE=.\possion.cpp
# End Source File
# Begin Source File

SOURCE=.\randop.cpp
# End Source File
# Begin Source File

SOURCE=.\ReadTxtData.cpp
# End Source File
# Begin Source File

SOURCE=.\showstr.cpp
# End Source File
# Begin Source File

SOURCE=.\test.cpp
# End Source File
# Begin Source File

SOURCE=.\test10.cpp
# End Source File
# Begin Source File

SOURCE=.\test11.cpp
# End Source File
# Begin Source File

SOURCE=.\test12.cpp
# End Source File
# Begin Source File

SOURCE=.\test13.cpp
# End Source File
# Begin Source File

SOURCE=.\test14.cpp
# End Source File
# Begin Source File

SOURCE=.\test15.cpp
# End Source File
# Begin Source File

SOURCE=.\test16.cpp
# End Source File
# Begin Source File

SOURCE=.\test17.cpp
# End Source File
# Begin Source File

SOURCE=.\test18.cpp
# End Source File
# Begin Source File

SOURCE=.\test19.cpp
# End Source File
# Begin Source File

SOURCE=.\test2.cpp
# End Source File
# Begin Source File

SOURCE=.\test20.cpp
# End Source File
# Begin Source File

SOURCE=.\test21.cpp
# End Source File
# Begin Source File

SOURCE=.\test3.cpp
# End Source File
# Begin Source File

SOURCE=.\test4.cpp
# End Source File
# Begin Source File

SOURCE=.\test5.cpp
# End Source File
# Begin Source File

SOURCE=.\test6.cpp
# End Source File
# Begin Source File

SOURCE=.\test7.cpp
# End Source File
# Begin Source File

SOURCE=.\test8.cpp
# End Source File
# Begin Source File

SOURCE=.\test9.cpp
# End Source File
# End Target
# End Project
