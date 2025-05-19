const images = document.querySelectorAll("#slideshow img");
const dots = document.querySelectorAll(".dot");
let currentIndex = 0;
const delay = 2500; // 25 seconds
let interval;

function updateSlidePosition() {
  images.forEach((img, idx) => {
    img.classList.remove("active");
    if (idx === currentIndex) {
      img.classList.add("active");
    }
  });

  dots.forEach(dot => dot.classList.remove("active"));
  dots[currentIndex].classList.add("active");
}

function moveToSlide(index) {
  currentIndex = index;
  updateSlidePosition();
  resetInterval();
}

function nextSlide() {
  currentIndex = (currentIndex + 1) % images.length;
  updateSlidePosition();
}

function resetInterval() {
  clearInterval(interval);
  interval = setInterval(nextSlide, delay);
}

// Initialize
updateSlidePosition();
interval = setInterval(nextSlide, delay);

const slideshow = document.querySelector('.slideshow-container');
const dashboardContent = document.querySelector('.dashboard-content');

dashboardContent.addEventListener('scroll', () => {
  const fadeThreshold = 60;
  slideshow.classList.toggle('slideshow-fade', dashboardContent.scrollTop > fadeThreshold);
});

document.addEventListener('DOMContentLoaded', () => {
  const dashboardCards = document.querySelectorAll('.dashboard-card');

  dashboardCards.forEach(card => {
    card.addEventListener('click', () => {
      // Hide dashboard section when a card is clicked
      const dashboardSection = document.querySelector('.dashboard-section');
      if (dashboardSection) {
        dashboardSection.style.display = 'none';
      }
      // Also remove active state from dashboard nav button
      const dashboardNavBtn = document.querySelector(`.sidebar nav button[onclick*="dashboard"]`);
      if (dashboardNavBtn) {
        dashboardNavBtn.classList.remove('active');
      }
    });
  });
});

