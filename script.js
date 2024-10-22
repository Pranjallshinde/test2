// Handle form submission
document.getElementById('contact-form').addEventListener('submit', function(event) {
    event.preventDefault();
    alert('Thank you. We will get back to you soon!');
    this.reset(); // Reset form after submission
});
