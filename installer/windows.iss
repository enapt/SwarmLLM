; SwarmLLM Windows Installer
; Built with Inno Setup 6 — https://jrsoftware.org/isinfo.php
;
; Bundles three binaries:
;   swarmllm.exe        — launcher: detects GPU at runtime, runs gpu or cpu variant
;   swarmllm-gpu.exe    — Vulkan (local inference, all GPU vendors) + CUDA static (split inference, NVIDIA)
;   swarmllm-cpu.exe    — CPU-only fallback (works on any Windows PC)
;
; Usage from CI:
;   ISCC.exe /DAppVersion=0.1.0 /DBinDir=C:\path\to\bins installer\windows.iss

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#ifndef BinDir
  #define BinDir "..\installer-bin"
#endif

#define AppName "SwarmLLM"
#define AppPublisher "SwarmLLM"
#define AppURL "https://github.com/enapt/SwarmLLM"
#define AppExeName "swarmllm.exe"
#define AppDataDir "{userappdata}\SwarmLLM"

[Setup]
AppId={{E3A1F2C4-8B7D-4E9A-A2F6-C1D5E8B3F047}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={autopf}\SwarmLLM
DefaultGroupName={#AppName}
AllowNoIcons=yes
LicenseFile=..\LICENSE-MIT
OutputDir=Output
OutputBaseFilename=SwarmLLM-Setup
; SetupIconFile=installer/swarmllm.ico  ; uncomment when icon asset is added
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; Require Windows 10 or later (Vulkan + modern driver support)
MinVersion=10.0
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
; No admin required — installs per-user by default
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath"; Description: "Add SwarmLLM to PATH (recommended for command-line use)"; GroupDescription: "Additional options:"; Flags: checked

[Files]
; Launcher — always runs, detects GPU and picks the right binary
Source: "{#BinDir}\swarmllm.exe";     DestDir: "{app}"; Flags: ignoreversion

; GPU binary — Vulkan (NVIDIA/AMD/Intel local inference) + CUDA static (NVIDIA split inference)
Source: "{#BinDir}\swarmllm-gpu.exe"; DestDir: "{app}"; Flags: ignoreversion

; CPU binary — universal fallback, works on any Windows PC
Source: "{#BinDir}\swarmllm-cpu.exe"; DestDir: "{app}"; Flags: ignoreversion

; Config and docs
Source: "{#BinDir}\default.toml";     DestDir: "{app}"; Flags: ignoreversion onlyifdoesntexist
Source: "{#BinDir}\INSTALL.txt";      DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\SwarmLLM"; Filename: "{app}\{#AppExeName}"
Name: "{group}\SwarmLLM Dashboard"; Filename: "{app}\{#AppExeName}"; Parameters: "run"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\SwarmLLM"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; Add to PATH if user selected that option
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
  ValueData: "{olddata};{app}"; \
  Check: PathNotContains('{app}'); Tasks: addtopath

[Run]
; Offer to open the dashboard after install
Filename: "{app}\{#AppExeName}"; Parameters: "run"; Description: "Launch SwarmLLM"; \
  Flags: nowait postinstall skipifsilent; WorkingDir: "{app}"

[UninstallDelete]
; Clean up data dir on uninstall only if user confirms (leave models intact by default)
Type: dirifempty; Name: "{#AppDataDir}"

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%n\
SwarmLLM automatically selects the best backend for your hardware:%n%n\
  NVIDIA GPU    GPU-accelerated (Vulkan + CUDA)%n\
  AMD / Intel   GPU-accelerated local inference (Vulkan)%n\
  No GPU        CPU fallback (works everywhere)%n%n\
No CUDA Toolkit or special drivers are required — standard graphics%n\
drivers are all you need.

[Code]
// Check if a path is already in the user PATH env var
function PathNotContains(const Path: string): Boolean;
var
  CurrentPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', CurrentPath) then
    CurrentPath := '';
  Result := Pos(Lowercase(Path), Lowercase(CurrentPath)) = 0;
end;
