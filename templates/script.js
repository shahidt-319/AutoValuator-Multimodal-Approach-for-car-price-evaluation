  function showTab(tabId) {
      document.querySelectorAll('.form').forEach(form => form.classList.remove('active'));
      document.querySelectorAll('.tab-buttons button').forEach(btn => btn.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      event.target.classList.add('active');
    }