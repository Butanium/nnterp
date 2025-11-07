// Dynamically add source link to navbar
document.addEventListener('DOMContentLoaded', function() {
    // Find the source link in the sidebar
    const sourceLink = document.querySelector('a[href*="_sources"]');
    
    if (sourceLink) {
        const sourceUrl = sourceLink.getAttribute('href');
        
        // Find the navbar icon links container
        const navbarIconLinks = document.querySelector('.navbar-icon-links');
        
        if (navbarIconLinks) {
            // Avoid duplicating icons when navigating via sphinx inline routing
            if (navbarIconLinks.querySelector('[data-doc-source-icon="true"]')) {
                return;
            }
 
            const iconLink = document.createElement('a');
            iconLink.className = 'nav-link nav-icon-link';
            iconLink.href = sourceUrl;
            iconLink.title = 'View documentation source';
            iconLink.setAttribute('aria-label', 'View documentation source');
            iconLink.target = '_blank';
            iconLink.rel = 'noopener noreferrer';
            iconLink.dataset.docSourceIcon = 'true';

            const icon = document.createElement('i');
            icon.className = 'fa-regular fa-file-lines';
            iconLink.appendChild(icon);

            const themeSwitcher = navbarIconLinks.querySelector('.theme-switch');
            if (themeSwitcher) {
                navbarIconLinks.insertBefore(iconLink, themeSwitcher);
            } else {
                navbarIconLinks.appendChild(iconLink);
            }
        }

        const sourceSidebarSection = sourceLink.closest('.sidebar-secondary-item');
        if (sourceSidebarSection) {
            sourceSidebarSection.remove();
        }
    }
});

