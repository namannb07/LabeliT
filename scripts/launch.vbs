Set fso = CreateObject("Scripting.FileSystemObject")
currentDir = fso.GetParentFolderName(WScript.ScriptFullName)
repoDir = fso.GetParentFolderName(currentDir)
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = repoDir
sh.Run """" & repoDir & "\.venv\Scripts\pythonw.exe"" -m auto_annotator", 0, False
