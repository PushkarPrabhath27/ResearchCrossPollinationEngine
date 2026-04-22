# 🔧 Fix: Docker "Virtualization Support Not Detected" Error

## The Problem

Docker requires **hardware virtualization** (Intel VT-x or AMD-V) to run. This feature is often **disabled by default** in your computer's BIOS settings.

![Docker Virtualization Error](uploaded_image_1765185107049.png)

---

## 🛠️ Solution: Step-by-Step Guide

### Step 1: Check if Virtualization is Enabled (Quick Check)

Before going to BIOS, let's confirm virtualization is disabled:

1. Press **Ctrl + Shift + Esc** to open Task Manager
2. Click the **Performance** tab
3. Click **CPU** on the left
4. Look at the bottom right for **"Virtualization: Enabled"** or **"Virtualization: Disabled"**

If it says **"Disabled"**, continue to Step 2.

---

### Step 2: Restart Your Computer and Enter BIOS

Different computers have different keys to enter BIOS. Here's how:

#### Method A: From Windows Settings (Recommended)

1. Press **Windows Key**
2. Click **Settings** (gear icon)
3. Click **System** → **Recovery**
4. Under "Advanced startup", click **Restart now**
5. Your computer will restart to a blue screen
6. Click **Troubleshoot**
7. Click **Advanced options**
8. Click **UEFI Firmware Settings**
9. Click **Restart**
10. Your computer will boot into BIOS

#### Method B: Using Key During Startup

1. Completely shut down your computer (not restart)
2. Turn it on and immediately start pressing the BIOS key repeatedly

**Common BIOS Keys by Brand:**

| Brand | Key to Press |
|-------|--------------|
| **Dell** | F2 or F12 |
| **HP** | F10 or Esc |
| **Lenovo** | F1 or F2 |
| **ASUS** | F2 or Delete |
| **Acer** | F2 or Delete |
| **MSI** | Delete |
| **Samsung** | F2 |
| **Toshiba** | F2 or F12 |
| **Microsoft Surface** | Hold Volume Up + Power |

---

### Step 3: Find and Enable Virtualization in BIOS

Once in BIOS, you need to find the virtualization setting. It has different names:

**Look for these terms:**
- **Intel VT-x** (Intel processors)
- **Intel Virtualization Technology**
- **VT-x**
- **AMD-V** (AMD processors)
- **SVM Mode** (AMD processors)
- **Vanderpool**

**Where to look (common locations):**

#### For Intel Processors:
```
Advanced → CPU Configuration → Intel Virtualization Technology → ENABLED
```
OR
```
Security → Virtualization → Intel VT-x → ENABLED
```
OR
```
System Configuration → Virtualization Technology → ENABLED
```

#### For AMD Processors:
```
Advanced → CPU Configuration → SVM Mode → ENABLED
```
OR
```
Advanced → AMD SVM Technology → ENABLED
```

#### Brand-Specific Guides:

**Dell:**
1. Go to **Advanced** or **Virtualization Support**
2. Find **Virtualization**, **Enable Intel Virtualization Technology**, or **VT for Direct I/O**
3. Change to **Enabled**

**HP:**
1. Go to **Advanced** → **System Options** or **Device Configurations**
2. Find **Virtualization Technology (VTx)**
3. Check the box or select **Enabled**

**Lenovo:**
1. Go to **Security** → **Virtualization**
2. Enable **Intel Virtualization Technology**
3. Enable **Intel VT-d Feature** (if available)

**ASUS:**
1. Press **F7** for Advanced Mode
2. Go to **Advanced** → **CPU Configuration**
3. Find **Intel Virtualization Technology**
4. Set to **Enabled**

---

### Step 4: Save and Exit BIOS

1. Look for **Save & Exit** option (usually F10)
2. Confirm **Yes** to save changes
3. Your computer will restart

---

### Step 5: Enable Windows Features

After restarting, open **PowerShell as Administrator** and run these commands:

```powershell
# Enable Hyper-V (required for Docker)
dism.exe /online /enable-feature /featurename:Microsoft-Hyper-V-All /all /norestart

# Enable Virtual Machine Platform
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Enable WSL
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

Then restart your computer:
```powershell
shutdown /r /t 0
```

---

### Step 6: Update WSL

After restart, open **PowerShell as Administrator** again:

```powershell
# Update WSL to version 2
wsl --update

# Set WSL 2 as default
wsl --set-default-version 2
```

---

### Step 7: Start Docker Desktop

1. Open **Docker Desktop** from Start Menu
2. Wait 1-2 minutes for it to initialize
3. You should see "Docker Desktop is running" with no errors

---

## ❌ If You Still Have Issues

### Problem: "WSL 2 installation is incomplete"

Download and install the WSL 2 Linux kernel update:
1. Go to: https://aka.ms/wsl2kernel
2. Download and install "wsl_update_x64.msi"
3. Restart Docker Desktop

### Problem: Can't find virtualization in BIOS

Some older computers or certain laptop models don't support virtualization. In this case:

**Alternative: Run WITHOUT Docker**

You can still run the project without Docker by using SQLite instead of PostgreSQL:

1. Skip the `docker-compose up` step
2. The application will use local file-based storage instead

I'll update the guide if you need this alternative!

### Problem: Virtualization shows "Enabled" but Docker still fails

Try these:
1. Uninstall Docker Desktop completely
2. Restart computer
3. Reinstall Docker Desktop
4. Choose "Use WSL 2 instead of Hyper-V" during installation

---

## 📞 Still Need Help?

If you're stuck at any step, let me know:
1. What brand/model is your laptop?
2. Do you have Intel or AMD processor?
3. What do you see after Step 1 (Task Manager check)?

I'll provide brand-specific instructions!

---

*This guide applies to Windows 10 (version 1903+) and Windows 11*
