 document.addEventListener('DOMContentLoaded', function () {
    var form = document.getElementById('conversionForm');
    form.addEventListener('submit', function () {
        var loader = document.getElementById('loader');
        loader.style.display = 'block';
    });
});