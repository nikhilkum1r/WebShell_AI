/* ═══════════════════════════════════════════════════════════
   Main Application Entry Point
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize icons safely
    if (typeof feather !== 'undefined') {
        feather.replace();
    }

    // Check API health and populate stats
    checkHealth();
    fetchBenchmarks();

    // Set up auto-refresh for health every 30 seconds
    setInterval(checkHealth, 30000);

    console.log("🛡️ Webshell Detector v2 Frontend Initialized");
});
