
document.addEventListener('DOMContentLoaded', function() {
    // File upload visualization
    const fileInput = document.getElementById('document-upload');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name;
            if (fileName) {
                const fileLabel = this.previousElementSibling;
                if (fileLabel && fileLabel.tagName === 'LABEL') {
                    fileLabel.innerHTML = `<i class="fas fa-file-alt"></i> ${fileName}`;
                    fileLabel.classList.add('file-selected');
                }
            }
        });
    }

    // Animate case cards on the results page with staggered delay
    const caseCards = document.querySelectorAll('.case-card');
    caseCards.forEach((card, index) => {
        card.style.setProperty('--i', index);
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // Form validation
    const form = document.querySelector('form[action="/analyze"]');
    if (form) {
        form.addEventListener('submit', function(e) {
            const textInput = document.getElementById('legal-issue');
            const fileInput = document.getElementById('document-upload');
            
            if ((!textInput || textInput.value.trim() === '') && 
                (!fileInput || fileInput.files.length === 0)) {
                e.preventDefault();
                alert('Please either describe your legal issue or upload a relevant document.');
            }
        });
    }
});