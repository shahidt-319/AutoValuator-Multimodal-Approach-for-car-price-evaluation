// Switch Login/Register
function switchMode(mode) {
    document.getElementById('loginForm').style.display = (mode === 'login') ? 'block' : 'none';
    document.getElementById('registerForm').style.display = (mode === 'register') ? 'block' : 'none';
    document.getElementById('loginTab').classList.toggle('active-tab', mode === 'login');
    document.getElementById('registerTab').classList.toggle('active-tab', mode === 'register');
}

// Login Email/Phone Tabs
function switchLoginType(type) {
    if (type === 'email') {
        document.getElementById('loginEmailField').style.display = 'block';
        document.getElementById('loginPhoneField').style.display = 'none';
        document.getElementById('emailTabLogin').classList.add('active-tab');
        document.getElementById('phoneTabLogin').classList.remove('active-tab');
    } else {
        document.getElementById('loginEmailField').style.display = 'none';
        document.getElementById('loginPhoneField').style.display = 'block';
        document.getElementById('emailTabLogin').classList.remove('active-tab');
        document.getElementById('phoneTabLogin').classList.add('active-tab');
    }
}

// Register Email/Phone Tabs
function switchRegType(type) {
    if (type === 'email') {
        document.getElementById('regEmailFields').style.display = 'flex';
        document.getElementById('regPhoneFields').style.display = 'none';
    } else {
        document.getElementById('regEmailFields').style.display = 'none';
        document.getElementById('regPhoneFields').style.display = 'flex';
    }
    document.getElementById('emailTabReg').classList.toggle('active-tab', type === 'email');
    document.getElementById('phoneTabReg').classList.toggle('active-tab', type !== 'email');
}

// OTP Box Enable
function sendOtp(type) {
    let url = "/send_otp";
    let email = document.getElementById("emailInput").value;
    let phone = document.getElementById("phoneInput").value;
    let data = { type: type };
    if(type === 'email') data.email = email;
    if(type === 'phone') data.phone = phone;
    fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(data => {
        alert(data.message || "OTP sent!");
        if(type === 'email') {
            document.getElementById('otpInputEmail').disabled = false;
            document.getElementById('otpInputEmail').focus();
        }
        if(type === 'phone') {
            document.getElementById('otpInputPhone').disabled = false;
            document.getElementById('otpInputPhone').focus();
        }
    })
    .catch(err => console.error(err));
}

// Initialize defaults
switchLoginType('email');
switchRegType('email');
