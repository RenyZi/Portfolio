timeline = gsap.timeline({defaults: {duration: 1} });


//Defining our animations Header

timeline.from(".head",{y : '-100px', ease: 'bounce'})
        .from(".profile",{opacity: 0})
        .from(".nav-item", {opacity: 0,stagger: .5})
        .from("#home-cont",{opacity:0, ease:'power2in'})
        .from(".dynamic",{opacity: 0})
        .from(".texts", {opacity: 0})
        .from(".more",{opacity: 0})
        .from("#butt",{opacity: 0, stagger: .5})
        .from("#imagprofile",{
                x: '150px',
                ease: 'bounce',
                opacity: 0
        })
        .from("#colsocial",{x : '100px', opacity:0, ease: 'power2in'})

