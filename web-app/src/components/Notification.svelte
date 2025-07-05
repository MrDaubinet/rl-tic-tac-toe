<script lang="ts">
  // Svelte libraries
  import { scale, fade } from 'svelte/transition';
  export let message: string = '';
  export let duration: number = 2000;
  export let onClose: () => void = () => {};

  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  $: if (message) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      onClose();
    }, duration);
  }

  // Optional: clear timeout on destroy
  import { onDestroy } from 'svelte';
  onDestroy(() => clearTimeout(timeoutId));
</script>

<div 
  in:scale out:fade
  class="fixed inset-0 flex items-end justify-center px-4 py-6 pointer-events-none sm:p-6 sm:items-start sm:justify-start">
  <!--
    Notification panel, show/hide based on alert state.

    Entering: "transform ease-out duration-300 transition"
      From: "translate-y-2 opacity-0 sm:translate-y-0 sm:translate-x-2"
      To: "translate-y-0 opacity-100 sm:translate-x-0"
    Leaving: "transition ease-in duration-100"
      From: "opacity-100"
      To: "opacity-0"
  -->
  <div class="pr-10 text-right text-white pointer-events-auto">
    <div class="text-xl">
      {#if message}
        {message}
      {/if}
    </div> 
  </div>
</div> 